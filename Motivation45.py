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

def move_expert_weights_to_device(model, layer_idx, expert_idx, device):
    """
    仅移动 (layer_idx, expert_idx) Expert 的 w1, w2, w3 参数至 `device`。
    Gate、LayerNorm 等其它子模块常驻 GPU，不做 Swap。
    """
    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
    try:
        expert_module = model.get_submodule(prefix)
    except AttributeError as e:
        print(f"[move_expert_weights_to_device] 未找到 {prefix}: {e}")
        return False

    count_moved = 0
    for name, param in expert_module.named_parameters():
        # 仅移动 w1, w2, w3 参数
        if any(w in name for w in ("w1", "w2", "w3")):
            param.data = param.data.to(device)
            count_moved += 1

    if count_moved > 0:
        print(f"[move_expert_weights_to_device] {prefix} 的 w1/w2/w3 => {device}")
        return True
    else:
        print(f"[move_expert_weights_to_device] {prefix} 未发现 w1/w2/w3")
        return False

def swap_in_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu):
    """
    将 (layer_idx, expert_idx) 的 w1/w2/w3 移动到 GPU，并更新映射 & 集合。
    返回 (swap_in_count, swap_out_count, swap_latency)
    """
    start_t = time.perf_counter()
    moved = move_expert_weights_to_device(model, layer_idx, expert_idx, "cuda")
    end_t = time.perf_counter()
    if moved:
        expert_device_map[(layer_idx, expert_idx)] = "cuda"
        experts_in_gpu.add((layer_idx, expert_idx))
        latency = end_t - start_t
        print(f"[swap_in_expert] Expert({layer_idx}, {expert_idx}) => GPU(w1,w2,w3)")
        return 1, 0, latency
    return 0, 0, 0.0

def swap_out_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu):
    """
    将 (layer_idx, expert_idx) 的 w1/w2/w3 移动到 CPU，并更新映射 & 集合。
    返回 (swap_in_count, swap_out_count, swap_latency)
    """
    start_t = time.perf_counter()
    moved = move_expert_weights_to_device(model, layer_idx, expert_idx, "cpu")
    end_t = time.perf_counter()
    if moved:
        expert_device_map[(layer_idx, expert_idx)] = "cpu"
        if (layer_idx, expert_idx) in experts_in_gpu:
            experts_in_gpu.remove((layer_idx, expert_idx))
        latency = end_t - start_t
        print(f"[swap_out_expert] Expert({layer_idx}, {expert_idx}) => CPU(w1,w2,w3)")
        return 0, 1, latency
    return 0, 0, 0.0

def find_idle_expert_for_swap_out(experts_in_gpu, needed_experts):
    """
    在 experts_in_gpu 中找不在 needed_experts 内的 expert 进行 swap out。
    若没有空闲 expert，则随机选择一位。
    """
    idle_experts = [ex for ex in experts_in_gpu if ex not in needed_experts]
    if idle_experts:
        return random.choice(idle_experts)
    else:
        return random.choice(list(experts_in_gpu))

def main():
    try:
        print("[main] 初始化加载模型...")
        if not torch.cuda.is_available():
            print("[main] 无可用 GPU, 退出。")
            return

        # 打印 GPU 信息
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            print(f"[main] GPU {i}: {prop.name}, total_mem={prop.total_memory/(1024**3):.2f} GB")

        # 登录
        token = os.getenv("HUGGINGFACE_TOKEN", "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")
        if not token:
            print("[main] HUGGINGFACE_TOKEN 未设置, 退出。")
            return
        login(token)

        # 4-bit 量化配置
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

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )

        print("[main] 将模型基础结构移动到 GPU...")
        model.to("cuda")

        print("\n[main] 初始化后设备分配:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")

        print("\n[main] 模型子模块:")
        for name, module in model.named_modules():
            print(f"  {name}")

        model.eval()
        print("[main] 模型加载完成！")

    except Exception as e:
        print(f"[main] 初始化阶段出错: {e}")
        traceback.print_exc()
        return

    # 初始化：将第 5~31 层 Expert 的 w1/w2/w3 移到 CPU；0~4 层保留在 GPU
    print("\n[main] 初始化: 仅把5~31层的 Experts (w1,w2,w3) => CPU, 保留 Gate/LN 在 GPU")
    num_layers_total = 32
    num_experts_per_layer = 8
    expert_device_map = {}
    experts_in_gpu = set()

    # 假设 0~4 层 Experts 仍在 GPU
    for layer_idx in range(0, 5):
        for expert_idx in range(num_experts_per_layer):
            expert_device_map[(layer_idx, expert_idx)] = "cuda"
            experts_in_gpu.add((layer_idx, expert_idx))

    # 将 5~31层 Experts 移到 CPU (仅 w1/w2/w3)
    for layer_idx in range(5, num_layers_total):
        for expert_idx in range(num_experts_per_layer):
            move_expert_weights_to_device(model, layer_idx, expert_idx, "cpu")
            expert_device_map[(layer_idx, expert_idx)] = "cpu"

    print("[main] 初始化完成, 开始处理请求...")

    try:
        # 示例 requests
        requests = [
            "What is artificial intelligence? It's about machine reasoning."
        ]*11
        # 若您有文件 requests.txt，可自行读取

        max_experts_in_gpu = 20
        total_swap_in_count  = 0
        total_swap_out_count = 0
        total_swap_latency   = 0.0

        request_lengths = []
        total_times     = []
        moe_latencies   = []

        for idx, req in enumerate(requests):
            print(f"\n[main] 第 {idx+1}/{len(requests)} 个请求: {req}")
            inputs = tokenizer(req, return_tensors="pt").to("cuda")
            req_length = inputs["input_ids"].shape[1]
            request_lengths.append(req_length)

            # (可选) 路由
            used_experts = {(6,0),(6,1)}  # 示例: 需要 6层 experts=0,1

            # swap_in needed experts
            for (l,e) in used_experts:
                if expert_device_map.get((l,e),"cpu") != "cuda":
                    # 若 GPU满 => swap_out一个空闲
                    if len(experts_in_gpu) >= max_experts_in_gpu:
                        idle_exp = find_idle_expert_for_swap_out(experts_in_gpu, used_experts)
                        print(f"[main] GPU已满, swap out idle {idle_exp}")
                        _, out_cnt, out_lat = swap_out_expert(model, idle_exp[0], idle_exp[1], expert_device_map, experts_in_gpu)
                        total_swap_out_count += out_cnt
                        total_swap_latency   += out_lat
                    in_in, in_out, lat_in = swap_in_expert(model, l,e, expert_device_map, experts_in_gpu)
                    total_swap_in_count  += in_in
                    total_swap_latency   += lat_in

            # 推理
            total_start = torch.cuda.Event(enable_timing=True)
            total_end   = torch.cuda.Event(enable_timing=True)
            total_start.record()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False
                )
            total_end.record()
            total_end.synchronize()
            rt_ms = total_start.elapsed_time(total_end)
            rt_s  = rt_ms/1000.0
            total_times.append(rt_s)

            # 假设 MoE 占一半
            moe_latency = rt_s*0.5
            moe_latencies.append(moe_latency)

            print(f"[main] MoE层时间(假设): {moe_latency:.4f}s, 总推理: {rt_s:.4f}s")

            # (可选) swap_out used_experts
            for (l,e) in used_experts:
                if (l,e) in experts_in_gpu:
                    _, out_cnt, out_lat = swap_out_expert(model, l,e, expert_device_map, experts_in_gpu)
                    total_swap_out_count += out_cnt
                    total_swap_latency   += out_lat

        # 对比图
        if len(request_lengths)>0:
            try:
                plt.figure(figsize=(10,6))
                plt.plot(request_lengths, moe_latencies, label='MoE Latency', marker='o')
                plt.plot(request_lengths, total_times, label='Total Inference Time', marker='^')
                plt.xlabel("Request Length (tokens)")
                plt.ylabel("Runtime (seconds)")
                plt.title("Expert-level Swap (w1,w2,w3 only), Gate/LN in GPU")
                plt.legend()
                plt.grid(True)
                plt.savefig("moe_swap_expert_weights_only.png")
                plt.show()
            except Exception as e:
                print(f"[main] 绘图时异常: {e}")
                traceback.print_exc()

        print(f"[main] 总 swap_in={total_swap_in_count}, swap_out={total_swap_out_count}, swap_latency={total_swap_latency:.4f}s")

    except Exception as e:
        print(f"[main] 请求处理阶段出错: {e}")
        traceback.print_exc()

if __name__=="__main__":
    torch.cuda.empty_cache()
    main()
