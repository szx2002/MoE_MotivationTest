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
    路径: model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}
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
    找到一个“空闲”的专家：即在 GPU 上，但在当前请求中不需要 (layer_idx, expert_idx)。
    如果没有空闲专家，则随机策略。
    """
    idle_experts = [ex for ex in experts_in_gpu if ex not in needed_experts]
    if idle_experts:
        return random.choice(idle_experts)
    else:
        # 若没有空闲专家，则随机
        return random.choice(list(experts_in_gpu))

def main():
    try:
        print("[main] 开始加载模型...")

        if not torch.cuda.is_available():
            print("[main] 没有可用的 GPU，退出。")
            return

        print("[main] 检查 GPU...")
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {prop.name}, total_mem={prop.total_memory/(1024**3):.2f} GB")

        # 登录 Hugging Face
        token = os.getenv("HUGGINGFACE_TOKEN", "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")
        if not token:
            print("[main] 请设置 HUGGINGFACE_TOKEN")
            return
        login(token)

        # 配置 4-bit 量化
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model_id = "mistralai/Mixtral-8x7B-v0.1"
        print(f"[main] 加载模型 {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )

        print("[main] 将基础模型移动到 GPU...")
        model.to("cuda")

        # 打印模型参数的设备情况
        print("\n[main] 初始化后设备分配:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")

        print("\n[main] 模型的所有子模块:")
        for name, module in model.named_modules():
            print(f"  {name}")

        model.eval()
        print("[main] 模型加载完成！")
    except Exception as e:
        print(f"[main] 错误在模型加载阶段: {e}")
        traceback.print_exc()
        return

    print("\n[main] 在初始化阶段，将特定 expert 移动到 CPU (示例: 第5~31层, experts 0~7等)")
    # 您可根据实际需求决定：只在 5~31 层做 swap？
    # 如果需要对embedding或其他sub-layer固定gpu/固定cpu，也在此指定
    expert_device_map = {}
    experts_in_gpu = set()
    # 假设: 每层 0~7 共 8 个 expert
    # 先将 0~4 层保留在 GPU？(如有需要)；将 5~31层(共27层)移动到CPU
    num_layers_total = 32
    num_experts_per_layer = 8

    # init: 0~4 层 + experts 0~7 全部在 GPU
    for layer_idx in range(0, 5):
        for expert_idx in range(num_experts_per_layer):
            expert_device_map[(layer_idx, expert_idx)] = "cuda"
            experts_in_gpu.add((layer_idx, expert_idx))

    # 将 5~31层移到 CPU
    for layer_idx in range(5, num_layers_total):
        for expert_idx in range(num_experts_per_layer):
            # 只要找到 prefix, 就 .to("cpu")
            move_expert_to_device(model, layer_idx, expert_idx, "cpu", expert_device_map)

    print("[main] 初始化阶段完成，开始请求处理...")

    try:
        # 准备请求
        input_file = "requests.txt"
        requests = []
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    req = line.strip()
                    if req:
                        requests.append(req)
        else:
            print(f"[main] 未找到请求文件 {input_file}，退出。")
            return

        if not requests:
            print("[main] 请求列表为空，退出。")
            return

        # GPU 同时保留的 expert 上限
        max_experts_in_gpu = 20

        total_swap_in_count = 0
        total_swap_out_count = 0
        total_swap_latency = 0.0

        request_lengths = []
        total_times = []
        moe_latencies = []

        for req_idx, req in enumerate(requests):
            print(f"\n[main] 处理第 {req_idx+1}/{len(requests)} 个请求: {req}")
            inputs = tokenizer(req, return_tensors="pt").to("cuda")
            req_length = inputs["input_ids"].shape[1]
            request_lengths.append(req_length)

            # ### 两步路由逻辑(如您需要):
            # Step 1: 获取 router_logits -> 确定需要激活的 (layer_idx, expert_idx)
            # 这里仅演示, 不做真实路由:
            used_experts = set()
            # 假设 router 只需要 6 层 experts 0,1
            used_experts.add((6, 0))
            used_experts.add((6, 1))
            # 如果您实际 router_logits 形状: [num_layers, batch_size, num_experts],
            #   => 做 argmax => (layer_idx, expert_idx)...

            # Step 2: 确保 used_experts 在 GPU
            #   如果 GPU 满了，就先 swap out 空闲expert
            for (l_idx, e_idx) in used_experts:
                current_dev = expert_device_map.get((l_idx, e_idx), "cpu")
                if current_dev != "cuda":
                    # 如果 GPU 满
                    if len(experts_in_gpu) >= max_experts_in_gpu:
                        # 找空闲(不在 used_experts 内)的expert
                        idle_expert = find_idle_expert_for_swap_out(experts_in_gpu, used_experts)
                        print(f"[main] GPU已满，将空闲expert {idle_expert}移到CPU")
                        out_in, out_out, lat_out = swap_out_expert(model, idle_expert[0], idle_expert[1], expert_device_map, experts_in_gpu)
                        total_swap_out_count += out_out
                        total_swap_latency += lat_out

                    # swap in
                    print(f"[main] swap in {l_idx}, expert={e_idx}")
                    in_in, in_out, lat_in = swap_in_expert(model, l_idx, e_idx, expert_device_map, experts_in_gpu)
                    total_swap_in_count += in_in
                    total_swap_latency += lat_in

            # Step 3: 执行推理
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,        # => userWarning
                    temperature=0.7,        # => userWarning: 只在 do_sample=True 时起作用
                    top_p=0.95,             # => userWarning
                    repetition_penalty=1.1,
                    use_cache=False
                )
            total_end.record()
            total_end.synchronize()

            runtime_ms = total_start.elapsed_time(total_end)
            runtime_s = runtime_ms / 1000.0
            total_times.append(runtime_s)

            # 假设 MoE 层占一半时间
            moe_latency = runtime_s * 0.5
            moe_latencies.append(moe_latency)

            print(f"[main] MoE层运行时间(假设值): {moe_latency:.4f}s, 整体时间: {runtime_s:.4f}s")

            # (可选) Step 4: swap out不需要的expert
            #   如果请求结束后, 6层 experts 0,1 不再需要 => swap out
            for (l_idx, e_idx) in used_experts:
                # 演示: 立即 swap out
                # 真实情况: 可能下个请求也需要 => 可做缓存策略
                if (l_idx, e_idx) in experts_in_gpu:
                    out_in, out_out, lat_out = swap_out_expert(
                        model, l_idx, e_idx, expert_device_map, experts_in_gpu
                    )
                    total_swap_out_count += out_out
                    total_swap_latency += lat_out

    except Exception as e:
        print(f"[main] 错误在请求处理阶段: {e}")
        traceback.print_exc()
        return

    try:
        if len(request_lengths) > 0:
            print("\n[main] 绘制对比图：MoE latency vs Total Runtime...")
            plt.figure(figsize=(10, 6))
            plt.plot(request_lengths, moe_latencies, label='MoE Latency', marker='o')
            plt.plot(request_lengths, total_times, label='Total Inference Runtime', marker='^')
            plt.xlabel("Request Length (number of tokens)")
            plt.ylabel("Runtime (seconds)")
            plt.title("MoE Latency vs Total Inference Runtime (Expert-level Swap)")
            plt.legend()
            plt.grid(True)
            plt.savefig("moe_vs_total_runtime_expert.png")
            plt.show()
    except Exception as e:
        print(f"[main] 绘图错误: {e}")
        traceback.print_exc()

    print(f"\n[main] 共计 swap in: {total_swap_in_count}, swap out: {total_swap_out_count}, swap latency总和: {total_swap_latency:.4f}s")

def find_idle_expert_for_swap_out(experts_in_gpu, needed_experts):
    """
    找到空闲的 expert: 不在 needed_experts 内的那些.
    如果无空闲, 则随机.
    """
    idle_experts = [ex for ex in experts_in_gpu if ex not in needed_experts]
    if idle_experts:
        return random.choice(idle_experts)
    else:
        return random.choice(list(experts_in_gpu))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
