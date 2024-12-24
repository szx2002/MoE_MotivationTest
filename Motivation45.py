# 文件示例: inference_moe_scheme_b.py

import os
import time
import random
import traceback
import matplotlib.pyplot as plt

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import login


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
        print(f"[swap_in_expert] Expert({layer_idx}, {expert_idx}) => GPU")
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
        print(f"[swap_out_expert] Expert({layer_idx}, {expert_idx}) => CPU")
        return 1, 1, latency
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
    #---------------------------------------------------------
    # 1) 初始化/加载模型
    #---------------------------------------------------------
    try:
        token = os.getenv("HUGGINGFACE_TOKEN", "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")  # 请替换为你的真实token，或设置环境变量
        if not token:
            print("[main] 未发现HUGGINGFACE_TOKEN，请自行设置后再试。")
        login(token)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model_id = "mistralai/Mixtral-8x7B-v0.1"  # 示例ID
        print(f"[main] 开始加载模型 {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            token=token,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        print("[main] 将模型移动到 GPU...")
        model.to("cuda")
        model.eval()
        print("[main] 模型加载完成！")

    except Exception as e:
        print("[main] 加载模型出错:", e)
        traceback.print_exc()
        return

    #---------------------------------------------------------
    # 2) 设置初始专家分布 (部分专家留在GPU，其余在CPU)
    #   假设我们知道这个模型有 32 层，每层 8 个 Experts
    #---------------------------------------------------------
    num_layers_total = 32
    num_experts_per_layer = 8

    experts_in_gpu = set()         # 用来记录当前在 GPU 的 (layer_idx, expert_idx)
    expert_device_map = {}         # 记录所有 (layer_idx, expert_idx) 当前所在设备

    # 将 0~4 层的所有 Experts 常驻 GPU，5~31 层移到 CPU
    for layer_idx in range(16, num_layers_total):
        for expert_idx in range(num_experts_per_layer):
            move_expert_weights_to_device(model, layer_idx, expert_idx, "cpu")
            expert_device_map[(layer_idx, expert_idx)] = "cpu"

    for layer_idx in range(0, 16):
        for expert_idx in range(num_experts_per_layer):
            expert_device_map[(layer_idx, expert_idx)] = "cuda"
            experts_in_gpu.add((layer_idx, expert_idx))

    # GPU 上最多容纳多少个 Expert
    max_experts_in_gpu = 128

    # 统计 Swap 次数和延时
    total_swap_in_count  = 0
    total_swap_out_count = 0
    total_swap_latency   = 0.0

    #---------------------------------------------------------
    # 3) 构造一些请求
    #---------------------------------------------------------
    requests = [
        "What is artificial intelligence? It's about machine reasoning.",
        "Tell me a joke about science.",
        "Explain the concept of quantum entanglement."
    ]

    #---------------------------------------------------------
    # 4) 逐请求推理：先 "只算路由" -> Swap In -> 再正式generate
    #---------------------------------------------------------
    for req_idx, req_text in enumerate(requests):
        print(f"\n[main] 第 {req_idx+1}/{len(requests)} 个请求: {req_text}")
        inputs = tokenizer(req_text, return_tensors="pt").to("cuda")

        # 4.1) 先只获取路由 logits，不进专家
        #      这里要依赖于你在modeling_mixtral.py中增加的
        #      MixtralForCausalLM.forward_router_only(...) 方法。
        try:
            router_logits_tuple = model.forward_router_only(
                input_ids=inputs["input_ids"]
            )
        except Exception as e:
            print("[main] 获取 router_logits 时出错:", e)
            traceback.print_exc()
            continue

        if not router_logits_tuple:
            print("[main] router_logits_tuple 为空，本次不做 Swap。")
            used_experts = []
        else:
            # 4.2) 根据 router_logits 解析出本次会被路由选中的 experts
            used_experts = []
            for layer_idx, logits in enumerate(router_logits_tuple):
                # 假设 logits 形状: (batch*seq_len, num_experts)
                selected = logits.argmax(dim=-1).tolist()
                unique_experts_in_layer = set(selected)
                for e_id in unique_experts_in_layer:
                    used_experts.append((layer_idx, e_id))

            used_experts = list(set(used_experts))
            print(f"[main] 预测本次会激活 {len(used_experts)} 个 experts: {used_experts}")

        # 4.3) Swap In
        for (l,e) in used_experts:
            current_dev = expert_device_map.get((l,e), "cpu")
            if current_dev != "cuda":
                # 如果 GPU 满 => swap out 一个空闲专家
                if len(experts_in_gpu) >= max_experts_in_gpu:
                    idle_exp = find_idle_expert_for_swap_out(experts_in_gpu, used_experts)
                    _, out_cnt, out_lat = swap_out_expert(
                        model, idle_exp[0], idle_exp[1],
                        expert_device_map, experts_in_gpu
                    )
                    total_swap_out_count += out_cnt
                    total_swap_latency   += out_lat

                in_in, _, lat_in = swap_in_expert(
                    model, l, e,
                    expert_device_map, experts_in_gpu
                )
                total_swap_in_count  += in_in
                total_swap_latency   += lat_in

        # 4.4) 现在真正调用 model.generate(...) 做完整推理
        try:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end   = torch.cuda.Event(enable_timing=True)
            total_start.record()

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )

            total_end.record()
            total_end.synchronize()
            elapsed_ms = total_start.elapsed_time(total_end)
            elapsed_s  = elapsed_ms / 1000.0

            gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[main] 推理结果: {gen_text}")
            print(f"[main] 本次完整推理耗时: {elapsed_s:.4f}s")

        except Exception as e:
            print("[main] 推理阶段出错:", e)
            traceback.print_exc()

        for (l,e) in used_experts:
            if l >= 5 and (l,e) in experts_in_gpu:
                _, out_cnt, out_lat = swap_out_expert(
                    model, l, e,
                    expert_device_map, experts_in_gpu
                )
                total_swap_out_count += out_cnt
                total_swap_latency   += out_lat

    #---------------------------------------------------------
    # 5) 打印统计
    #---------------------------------------------------------
    print(f"\n[main] 所有请求完成!")
    print(f"       swap_in={total_swap_in_count}, swap_out={total_swap_out_count}, "
          f"swap_latency={total_swap_latency:.4f}s")


if __name__ == "__main__":
    main()
