import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from huggingface_hub import login
import matplotlib.pyplot as plt
import random
import traceback

def move_expert_to_device(model, layer_idx, expert_idx, device, expert_device_map):
    """
    将指定 layer_idx 下的某个 expert_idx 移动到 CPU 或 GPU。
    并更新 expert_device_map。
    => 路径: model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}
    """
    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
    try:
        expert_module = model.get_submodule(prefix)
        expert_module.to(device)
        expert_device_map[(layer_idx, expert_idx)] = device
        print(f"[move_expert_to_device] 已将 {prefix} 移动到 {device}")
        return True
    except AttributeError as e:
        print(f"[move_expert_to_device] 模型中未找到 {prefix}: {e}")
        return False

def swap_in_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu):
    """
    将指定的 expert 移动到 GPU，并更新 expert_device_map & experts_in_gpu。
    返回 (swap_in_count, swap_out_count, swap_latency)
    """
    start_t = time.perf_counter()
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cuda", expert_device_map)
    end_t = time.perf_counter()
    if moved:
        experts_in_gpu.add((layer_idx, expert_idx))
        latency = end_t - start_t
        print(f"[swap_in_expert] Expert (layer={layer_idx}, expert={expert_idx}) -> GPU")
        return 1, 0, latency
    return 0, 0, 0.0

def swap_out_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu):
    """
    将指定的 expert 移动到 CPU，并更新 expert_device_map & experts_in_gpu。
    返回 (swap_in_count, swap_out_count, swap_latency)
    """
    start_t = time.perf_counter()
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cpu", expert_device_map)
    end_t = time.perf_counter()
    if moved:
        if (layer_idx, expert_idx) in experts_in_gpu:
            experts_in_gpu.remove((layer_idx, expert_idx))
        latency = end_t - start_t
        print(f"[swap_out_expert] Expert (layer={layer_idx}, expert={expert_idx}) -> CPU")
        return 0, 1, latency
    return 0, 0, 0.0

def find_idle_expert_for_swap_out(experts_in_gpu, needed_experts):
    """
    找到空闲 expert: 即当前在 GPU 上，但此次请求不需要的 (layer_idx, expert_idx)。
    若没有空闲，就随机。
    """
    idle_experts = [ex for ex in experts_in_gpu if ex not in needed_experts]
    if idle_experts:
        return random.choice(idle_experts)
    else:
        return random.choice(list(experts_in_gpu))

def get_router_logits(model, inputs):
    """
    第一步: 仅forward一次, 获取 router_logits 形状 [num_layers, batch_size, num_experts].
    """
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_router_logits=True,
            return_dict=True,
            use_cache=False
        )
    return outputs.router_logits

def get_used_experts(router_logits):
    """
    根据 router_logits 判断本次请求需要的 experts。
    router_logits[i].argmax(dim=-1) => [batch_size] => for each sample => expert_idx
    """
    used_experts = set()
    # router_logits: tuple of length = num_layers
    # router_logits[i] => [batch_size, num_experts]
    for layer_idx, logits in enumerate(router_logits):
        top_experts = logits.argmax(dim=-1).tolist()  # shape [batch_size], => list
        for ex_idx in top_experts:
            used_experts.add((layer_idx, ex_idx))
    return used_experts

def ensure_experts_on_gpu(model, needed_experts, expert_device_map, experts_in_gpu, max_experts_in_gpu):
    """
    对 needed_experts 做 swap in. 若 GPU 满, 优先 swap out 不在 needed_experts 中的 idle experts。
    """
    for (layer_idx, expert_idx) in needed_experts:
        current_dev = expert_device_map.get((layer_idx, expert_idx), "cpu")
        if current_dev != "cuda":
            # GPU已满 => swap out某空闲expert
            if len(experts_in_gpu) >= max_experts_in_gpu:
                idle = find_idle_expert_for_swap_out(experts_in_gpu, needed_experts)
                print(f"  [ensure_experts_on_gpu] GPU已满, swap out 空闲expert {idle}")
                out_in, out_out, lat_out = swap_out_expert(model, idle[0], idle[1], expert_device_map, experts_in_gpu)
            print(f"  [ensure_experts_on_gpu] swap in => (layer={layer_idx}, expert={expert_idx})")
            in_in, in_out, lat_in = swap_in_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu)

def main():
    try:
        print("[main] 开始加载模型...")

        if not torch.cuda.is_available():
            print("[main] 无可用 GPU, 退出.")
            return

        # 检查GPU
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {prop.name}, total_mem={prop.total_memory/(1024**3):.2f} GB")

        token = os.getenv("HUGGINGFACE_TOKEN", "hf_XXX")
        if not token:
            print("[main] 未设置 Token.")
            return
        login(token)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model_id = "mistralai/Mixtral-8x7B-v0.1"
        print(f"[main] 加载模型 {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        print("[main] 将模型(基础组件)移动到GPU...")
        model.to("cuda")

        print("\n[main] 打印参数设备信息:")
        for name,param in model.named_parameters():
            print(f"  {name}: {param.device}")

        model.eval()
        print("[main] 模型加载完成.")

    except Exception as e:
        print(f"[main] 模型加载错误: {e}")
        traceback.print_exc()
        return

    try:
        # 初始化：将 5-31层的 experts 移到 CPU
        # => expert_device_map[(layer_idx, expert_idx)] = "cpu" or "cuda"
        expert_device_map = {}
        experts_in_gpu = set()
        num_layers_total = 32
        num_experts_per_layer = 8

        print("\n[main] 初始化: 将 layer=5..31 的 experts=0..7 移到 CPU")
        # 先默认 0~4 层所有 experts 在 GPU
        for l_idx in range(0, 5):
            for ex_idx in range(num_experts_per_layer):
                expert_device_map[(l_idx, ex_idx)] = "cuda"
                experts_in_gpu.add((l_idx, ex_idx))

        # 其余 5~31 层 => CPU
        for l_idx in range(5, num_layers_total):
            for ex_idx in range(num_experts_per_layer):
                move_expert_to_device(model, l_idx, ex_idx, "cpu", expert_device_map)
        print("[main] 初始化完成.")

        # 加载请求
        input_file = "requests.txt"
        requests = []
        if not os.path.exists(input_file):
            print(f"[main] 未找到请求文件 {input_file}, 退出.")
            return
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    requests.append(line)
        if not requests:
            print("[main] 请求列表为空, 退出.")
            return

        max_experts_in_gpu = 40  # GPU上最多可容纳多少 experts
        total_swap_in_count = 0
        total_swap_out_count = 0
        total_swap_latency = 0.0

        request_lengths = []
        total_times = []
        moe_latencies = []

        for i, req in enumerate(requests):
            print(f"\n[main] 处理第 {i+1}/{len(requests)} 个请求: {req}")
            # 先tokenize
            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            # 第一步: 获取 router_logits => used_experts
            try:
                router_logits = get_router_logits(model, inputs)
            except Exception as re:
                print(f"[main] 获取 router_logits 时出错: {re}")
                traceback.print_exc()
                continue

            if not router_logits or len(router_logits) == 0:
                print("[main] router_logits 为空, 跳过请求")
                continue
            used_experts = get_used_experts(router_logits)
            print(f"  [main] 需要 experts: {used_experts}")

            # 第二步: swap_in 必要 experts
            ensure_experts_on_gpu(model, used_experts, expert_device_map, experts_in_gpu, max_experts_in_gpu)

            # 第三步: 执行生成
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        num_return_sequences=1,
                        do_sample=False,
                        temperature=0.7,   # 警告: do_sample=False
                        top_p=0.95,        # 警告: do_sample=False
                        repetition_penalty=1.1,
                        use_cache=False
                    )
            except Exception as e_gen:
                print(f"[main] generate出错: {e_gen}")
                # 打印当前 experts 设备信息
                for (k,v) in expert_device_map.items():
                    print(f"    Expert {k} -> {v}")
                traceback.print_exc()
                continue
            total_end.record()
            total_end.synchronize()

            runtime_ms = total_start.elapsed_time(total_end)
            runtime_s = runtime_ms/1000.0
            total_times.append(runtime_s)
            # 假设 MoE层消耗一半时间
            moe_latency = runtime_s*0.5
            moe_latencies.append(moe_latency)

            # (可选) Swap out 不再需要的 experts
            # 例如: 这次请求结束后, used_experts都可以 swap_out, 也可部分保留
            for ex in used_experts:
                if ex in experts_in_gpu:
                    # do swap out
                    _, out_cnt, lat = swap_out_expert(model, ex[0], ex[1], expert_device_map, experts_in_gpu)
                    total_swap_out_count += out_cnt
                    total_swap_latency += lat

            print(f"  [main] runtime={runtime_s:.4f}s, MoELatency={moe_latency:.4f}s")

    except Exception as e2:
        print(f"[main] 错误在请求处理阶段: {e2}")
        traceback.print_exc()
        return

    # 绘图
    if request_lengths:
        plt.figure()
        plt.plot(request_lengths, moe_latencies, label="MoE Latency", marker='o')
        plt.plot(request_lengths, total_times, label="Total Runtime", marker='^')
        plt.xlabel("Request length (#tokens)")
        plt.ylabel("Time(s)")
        plt.title("MoE Expert-level Swap In/Out")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_expert_swap.png")
        plt.show()

    print(f"\n[main] 全部请求完毕. Swap统计: swap_in={total_swap_in_count}, swap_out={total_swap_out_count}, swap_latency={total_swap_latency:.4f}s")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
