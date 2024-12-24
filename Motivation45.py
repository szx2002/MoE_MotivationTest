import os
import time
import random
import traceback
import torch
import matplotlib.pyplot as plt

# 假设你在 modeling_mixtral.py 中已经定义了 global_swap_monitor 实例
from modeling_mixtral import global_swap_monitor

# 假设已有 BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, login, etc.
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


def move_expert_weights_to_device(model, layer_idx, expert_idx, device):
    """
    仅移动 (layer_idx, expert_idx) 的 w1, w2, w3 参数至 `device`。
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
    这里新增对 global_swap_monitor 的记录。
    """
    start_t = time.perf_counter()
    moved = move_expert_weights_to_device(model, layer_idx, expert_idx, "cuda")
    end_t = time.perf_counter()

    if moved:
        expert_device_map[(layer_idx, expert_idx)] = "cuda"
        experts_in_gpu.add((layer_idx, expert_idx))
        latency = end_t - start_t

        # 监测：在 global_swap_monitor 中累加 Swap In 次数 & 延时
        global_swap_monitor.add_swap_in(latency)

        print(f"[swap_in_expert] Expert({layer_idx}, {expert_idx}) => GPU, latency={latency:.4f}s")
        return 1, 0, latency

    return 0, 0, 0.0


def swap_out_expert(model, layer_idx, expert_idx, expert_device_map, experts_in_gpu):
    """
    将 (layer_idx, expert_idx) 的 w1/w2/w3 移动到 CPU，并更新映射 & 集合。
    这里新增对 global_swap_monitor 的记录。
    """
    start_t = time.perf_counter()
    moved = move_expert_weights_to_device(model, layer_idx, expert_idx, "cpu")
    end_t = time.perf_counter()

    if moved:
        if (layer_idx, expert_idx) in experts_in_gpu:
            experts_in_gpu.remove((layer_idx, expert_idx))
        expert_device_map[(layer_idx, expert_idx)] = "cpu"
        latency = end_t - start_t

        # 监测：在 global_swap_monitor 中累加 Swap Out 次数 & 延时
        global_swap_monitor.add_swap_out(latency)

        print(f"[swap_out_expert] Expert({layer_idx}, {expert_idx}) => CPU, latency={latency:.4f}s")
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
    # -------------------------------
    # 1) 初始化/加载模型 & 分词器
    # -------------------------------
    try:
        token = os.getenv("HUGGINGFACE_TOKEN", "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")
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

        model.to("cuda")
        model.eval()
        print("[main] 模型加载完成！")

    except Exception as e:
        print("[main] 加载模型出错:", e)
        traceback.print_exc()
        return

    # -------------------------------
    # 2) 初始化专家 CPU/GPU 分布
    # -------------------------------
    num_layers_total = 32
    num_experts_per_layer = 8
    experts_in_gpu = set()
    expert_device_map = {}

    # 例：保留 0~4 层在 GPU，其余在 CPU
    for layer_idx in range(16, num_layers_total):
        for expert_idx in range(num_experts_per_layer):
            move_expert_weights_to_device(model, layer_idx, expert_idx, "cpu")
            expert_device_map[(layer_idx, expert_idx)] = "cpu"

    for layer_idx in range(0, 16):
        for expert_idx in range(num_experts_per_layer):
            expert_device_map[(layer_idx, expert_idx)] = "cuda"
            experts_in_gpu.add((layer_idx, expert_idx))

    max_experts_in_gpu = 128

    # -------------------------------
    # 3) 构造一些请求
    # -------------------------------
    requests = [
        "What is artificial intelligence? It's about machine reasoning.",
        "Tell me a joke about science."
    ]

    # -------------------------------
    # 4) 处理请求
    # -------------------------------
    for req_idx, req_text in enumerate(requests):
        print(f"\n[main] 第 {req_idx+1}/{len(requests)} 个请求: {req_text}")

        # 重置 Swap 监测器，以便统计当次 request 的 Swap
        global_swap_monitor.reset()

        inputs = tokenizer(req_text, return_tensors="pt").to("cuda")

        # 4.1) （可选）先只获取 router_logits（若你实现了 forward_router_only）
        #      这里仅示例，不做 forward_router_only
        # used_experts = [...]
        # swap_in_expert(...) ...

        # 4.2) 直接推理
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("[main] 推理结果 =>", gen_text)

        except Exception as e:
            print("[main] 推理阶段出错:", e)
            traceback.print_exc()

        # 4.3) （可选）Swap Out 不需要的专家
        # 例如：保留 0~4层在 GPU，其它都移回 CPU
        # 这里仅做演示，全局都移回 CPU
        for (l, e) in list(experts_in_gpu):
            if l >= 32:
                swap_out_expert(model, l, e, expert_device_map, experts_in_gpu)

        # -------------------------------
        # 5) 打印该 request 的 Swap 统计
        # -------------------------------
        stats = global_swap_monitor.get_stats()
        print(
            f"\n[main] 本次请求 Swap统计: "
            f"swap_in_count={stats['swap_in_count']}, "
            f"swap_out_count={stats['swap_out_count']}, "
            f"swap_in_latency={stats['swap_in_latency']:.4f}s, "
            f"swap_out_latency={stats['swap_out_latency']:.4f}s"
        )

    print("\n[main] 所有请求处理完毕。")


if __name__ == "__main__":
    # 可选：清理 GPU 缓存
    torch.cuda.empty_cache()
    main()
