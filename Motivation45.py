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

def move_layer_to_device(model, layer_idx, device, layer_device_map):
    """
    将指定 Layer 及其所有参数移动到目标设备（CPU 或 GPU），并更新 layer_device_map。
    """
    layer_prefix = f"model.layers.{layer_idx}"
    try:
        layer_module = model.get_submodule(layer_prefix)
        layer_module.to(device)
        layer_device_map[layer_idx] = device
        print(f"[move_layer_to_device] 已将 {layer_prefix} 移动到 {device}")
        return True
    except AttributeError as e:
        print(f"[move_layer_to_device] 模型中未找到 {layer_prefix}: {e}")
        return False

def swap_in_layer(model, layer_idx, layer_device_map, layers_in_gpu):
    """
    将指定的 Layer 移动到 GPU，并更新 layer_device_map 和 layers_in_gpu 集合。
    """
    start_t = time.perf_counter()
    moved = move_layer_to_device(model, layer_idx, "cuda", layer_device_map)
    end_t = time.perf_counter()
    if moved:
        layers_in_gpu.add(layer_idx)
        swap_latency = end_t - start_t
        print(f"[swap_in_layer] Layer {layer_idx} 已被移动到 GPU")
        return 1, 0, swap_latency
    else:
        return 0, 0, 0.0

def swap_out_layer(model, layer_idx, layer_device_map, layers_in_gpu):
    """
    将指定的 Layer 移动到 CPU，并更新 layer_device_map 和 layers_in_gpu 集合。
    """
    start_t = time.perf_counter()
    moved = move_layer_to_device(model, layer_idx, "cpu", layer_device_map)
    end_t = time.perf_counter()
    if moved:
        if layer_idx in layers_in_gpu:
            layers_in_gpu.remove(layer_idx)
        swap_latency = end_t - start_t
        print(f"[swap_out_layer] Layer {layer_idx} 已被移动到 CPU")
        return 0, 1, swap_latency
    else:
        return 0, 0, 0.0

def find_idle_layer_for_swap_out(layers_in_gpu, current_used_layers):
    """
    找到空闲的 Layer 进行 swap out：即不在当前请求中需要使用的 Layer。
    如果都在使用中，可根据其他策略（如随机、LRU）做替换。
    """
    idle_layers = [idx for idx in layers_in_gpu if idx not in current_used_layers]
    if idle_layers:
        # 如果有空闲的 Layer，先移除它们
        return random.choice(idle_layers)
    else:
        # 若没有空闲 Layer，则随机策略
        return random.choice(list(layers_in_gpu))

def main():
    try:
        print("[main] 开始加载模型...")

        # 检查 GPU 可用性
        print("\n[main] 检查 GPU 状态...")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_prop = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {gpu_prop.name}, 显存: {gpu_prop.total_memory / (1024 ** 3):.2f} GB")
        else:
            print("[main] 没有可用的 GPU。")
            return

        # 使用环境变量管理 Token
        token = os.getenv("HUGGINGFACE_TOKEN", "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")
        if not token:
            print("[main] 请设置环境变量 HUGGINGFACE_TOKEN")
            return
        login(token)

        # 4-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 加载模型并将其移动到 GPU（只是将 embedding 等基础结构先放到 GPU）
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        print(f"[main] 加载模型 {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=True
        )

        # 加载分词器
        print("[main] 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )

        # 先将模型的 embedding 和 0~4 层放到 GPU 上
        print("\n[main] 初始化：将模型整体先移动到 GPU...")
        model.to("cuda")

        # 打印设备分配情况
        print("\n[main] 模型加载后设备分配:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")

        # 打印所有模块名称
        print("\n[main] 模型的所有模块名称:")
        for name, module in model.named_modules():
            print(f"  {name}")

        model.eval()
        print("[main] 模型加载完成！")
    except Exception as e:
        print(f"[main] 错误在模型加载阶段: {str(e)}")
        traceback.print_exc()
        return

    print("\n[main] 模型加载完成，开始处理请求...")

    try:
        # 准备请求
        input_file = "requests.txt"
        requests = []
        if os.path.exists(input_file):
            print(f"[main] 加载请求文件 '{input_file}'...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    req = line.strip()
                    if req:
                        requests.append(req)
        else:
            print(f"[main] 请求文件 '{input_file}' 不存在。")
            return

        if not requests:
            print("[main] 没有有效的请求。")
            return

        # 假设模型有 32 层
        num_layers = 32

        # 第 5 ~ 31 层统一移到 CPU
        # 并建立 layer_device_map，以记录每层所在的设备
        layer_device_map = {}
        # 先默认所有层在 GPU（0-31）
        for idx in range(num_layers):
            layer_device_map[idx] = "cuda"

        # 实际将 5~31 层移到 CPU
        print("\n[main] 初始化阶段：将第 5~31 层移动到 CPU...")
        for layer_idx in range(5, num_layers):
            move_layer_to_device(model, layer_idx, "cpu", layer_device_map)

        # 记录 GPU 中的层集合
        # 由于 0~4 层在 GPU，所以初始为 {0,1,2,3,4}
        layers_in_gpu = set(range(0, 5))

        # GPU 上最多保持 5 个 Layer
        max_layers_in_gpu = 5

        # 统计
        total_swap_in_count = 0
        total_swap_out_count = 0
        total_swap_latency = 0.0

        request_lengths = []
        total_times = []
        moe_latencies = []

        for req_idx, req in enumerate(requests):
            print(f"\n[main] 处理请求 {req_idx+1}/{len(requests)}: '{req}'")
            inputs = tokenizer(req, return_tensors="pt").to("cuda")
            req_length = inputs["input_ids"].shape[1]
            request_lengths.append(req_length)

            # 假设需要用到多少层，这里简化地说前 10 层都需要
            # 或者可以根据 router logits 确定
            needed_layers = list(range(0, 32))  # 示例
            # 将 needed_layers 中在 CPU 上的部分 swap in 到 GPU
            for l_idx in needed_layers:
                current_dev = layer_device_map.get(l_idx, "cuda")
                if current_dev != "cuda":
                    # GPU 满了，需要 swap out
                    if len(layers_in_gpu) >= max_layers_in_gpu:
                        # 找到空闲的 Layer（不在 needed_layers 的）
                        idle_layer = find_idle_layer_for_swap_out(layers_in_gpu, needed_layers)
                        print(f"  GPU已满，正在将空闲 Layer {idle_layer} 移动回 CPU")
                        out_in, out_out, lat_out = swap_out_layer(model, idle_layer, layer_device_map, layers_in_gpu)
                        total_swap_out_count += out_out
                        total_swap_latency += lat_out

                    print(f"[main] 将 Layer {l_idx} 移动到 GPU...")
                    in_in, in_out, lat_in = swap_in_layer(model, l_idx, layer_device_map, layers_in_gpu)
                    total_swap_in_count += in_in
                    total_swap_latency += lat_in

            # 使用 CUDA 事件计时
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)

            total_start.record()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False
                )
            total_end.record()
            total_end.synchronize()

            total_runtime_ms = total_start.elapsed_time(total_end)
            total_runtime = total_runtime_ms / 1000.0

            # 假设 MoE 层运行时间是总运行时间的一部分
            moe_latency = total_runtime * 0.5
            moe_latencies.append(moe_latency)
            total_times.append(total_runtime)

            print(f"[main] MoE层运行时间(假设值): {moe_latency:.4f}s")
            print(f"[main] 整个预测过程运行时间: {total_runtime:.4f}s")

            # Swap out 不需要的 Layer（示例逻辑）
            # 如果只需要 0~4 后续在 GPU，上述 needed_layers 不包含 5~9，就可以 swap out
            for l_idx in needed_layers:
                # 这里假设请求结束后，这些 needed_layers 又可以 swap out
                if l_idx >= 5:
                    # Swap out
                    if l_idx in layers_in_gpu:
                        out_in, out_out, lat_out = swap_out_layer(model, l_idx, layer_device_map, layers_in_gpu)
                        total_swap_out_count += out_out
                        total_swap_latency += lat_out

    except Exception as e:
        print(f"[main] 错误在请求处理阶段: {str(e)}")
        traceback.print_exc()
        return

    try:
        # 绘制对比图
        if len(request_lengths) > 0:
            print("\n[main] 绘制对比图：MoE latency vs Total Runtime...")
            plt.figure(figsize=(10, 6))
            plt.plot(request_lengths, moe_latencies, label='MoE Latency', marker='o')
            plt.plot(request_lengths, total_times, label='Total Inference Runtime', marker='^')
            plt.xlabel("Request Length (number of tokens)")
            plt.ylabel("Runtime (seconds)")
            plt.title("MoE Latency vs Total Inference Runtime (Partial Model)")
            plt.legend()
            plt.grid(True)
            plt.savefig("moe_vs_total_runtime_partial.png")
            plt.show()
    except Exception as e:
        print(f"[main] 错误在绘图阶段: {str(e)}")
        traceback.print_exc()

    print(f"\n[main] 所有请求合计：swap in次数={total_swap_in_count}, swap out次数={total_swap_out_count}, swap操作总延时={total_swap_latency:.4f}s")

def find_idle_layer_for_swap_out(layers_in_gpu, current_used_layers):
    """
    找到空闲的 Layer 进行 swap out：即不在当前请求中需要使用的 Layer。
    如果都在使用中，则随机策略。
    """
    idle_layers = [layer_idx for layer_idx in layers_in_gpu if layer_idx not in current_used_layers]
    if idle_layers:
        return random.choice(idle_layers)
    else:
        return random.choice(list(layers_in_gpu))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
