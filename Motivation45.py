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

    # 假设我们已知模型有 32 层，每层 8 个 Expert（示例，可根据实际情况调整）
    num_layers_total = 32
    num_experts_per_layer = 8

    # 初始化：将第 5~31 层 Expert 的 w1/w2/w3 移到 CPU；0~4 层保留在 GPU
    print("\n[main] 初始化: 仅把5~31层的 Experts (w1,w2,w3) => CPU, 保留 Gate/LN 在 GPU")
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
        ] * 5

        # 设定 GPU 上最多保留多少个 Experts
        max_experts_in_gpu = 40
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

            # -------------------------------------------------------
            # 1) 先做一次前向传播来获取 router_logits（动态检测 used_experts）
            # -------------------------------------------------------
            try:
                with torch.inference_mode():
                    router_outputs = model(
                        **inputs,
                        output_router_logits=True,
                        return_dict=True,
                        use_cache=False
                    )
            except Exception as e:
                print(f"[main] 获取 router_logits 时出错: {e}")
                traceback.print_exc()
                continue

            router_logits_tuple = router_outputs.router_logits
            if len(router_logits_tuple) == 0:
                # 若模型不输出router_logits，可能是MoE部分未启用
                print("  [main] 本次请求未产生 router_logits，跳过 Swap 逻辑...")
                used_experts = []
            else:
                # 每个 router_logits 的形状: (sequence_length, num_experts)
                # router_logits_tuple 的长度 = 实际 MoE 层数
                num_layers_actual = len(router_logits_tuple)
                print(f"  [main] router_logits_tuple 大小: {num_layers_actual} 个元素")

                # 按照第一份代码的做法，argmax(dim=-1)，获取每个 token 在该层被分配的 expert
                selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]
                # 堆叠后形状: (sequence_length, num_layers_actual)
                selected_experts = torch.stack(selected_experts, dim=-1)

                used_experts = []
                sequence_length, num_layers_actual = selected_experts.shape
                num_total_experts = 0

                for layer_idx in range(num_layers_actual):
                    experts_in_layer = selected_experts[:, layer_idx].tolist()
                    unique_experts_in_layer = set(experts_in_layer)
                    num_unique_experts = len(unique_experts_in_layer)
                    num_total_experts += num_unique_experts
                    print(f"    Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}")
                    for e_id in unique_experts_in_layer:
                        used_experts.append((layer_idx, e_id))

                used_experts = list(set(used_experts))
                print(f"  [main] 本次请求所有激活专家数(去重后) = {len(used_experts)}")

            # -------------------------------------------------------
            # 2) Swap In 需要的 Experts
            # -------------------------------------------------------
            for (l, e) in used_experts:
                current_device = expert_device_map.get((l, e), "cpu")
                if current_device != "cuda":
                    # 如果 GPU 已满，先 Swap Out 一个空闲的 Expert
                    if len(experts_in_gpu) >= max_experts_in_gpu:
                        idle_exp = find_idle_expert_for_swap_out(experts_in_gpu, used_experts)
                        print(f"[main] GPU已满, swap out idle {idle_exp}")
                        _, out_cnt, out_lat = swap_out_expert(
                            model, idle_exp[0], idle_exp[1],
                            expert_device_map, experts_in_gpu
                        )
                        total_swap_out_count += out_cnt
                        total_swap_latency   += out_lat

                    in_in, _, lat_in = swap_in_expert(model, l, e, expert_device_map, experts_in_gpu)
                    total_swap_in_count  += in_in
                    total_swap_latency   += lat_in

            # -------------------------------------------------------
            # 3) 进行真正的推理 (model.generate)
            # -------------------------------------------------------
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
            rt_s  = rt_ms / 1000.0
            total_times.append(rt_s)

            # 这里我们假设 MoE 占推理时间的一半
            moe_latency = rt_s * 0.5
            moe_latencies.append(moe_latency)

            print(f"[main] 推理完成, MoE层估计耗时: {moe_latency:.4f}s, 总推理耗时: {rt_s:.4f}s")

            # -------------------------------------------------------
            # 4)（可选）Swap Out 刚刚使用的 Experts
            #    如果你希望保持 GPU 专家数量不变或做分批管理，
            #    可根据策略将刚刚用过的专家也 Swap Out 回 CPU
            # -------------------------------------------------------
            # 如果你想让 GPU 上常驻 0~4层专家，那么这里只 Swap Out 5~31层
            # 下面仅做演示：
            for (l, e) in used_experts:
                # 如果你希望保持低显存占用，则可把用过的专家都移除
                if (l, e) in experts_in_gpu and l >= 5:
                    _, out_cnt, out_lat = swap_out_expert(
                        model, l, e,
                        expert_device_map, experts_in_gpu
                    )
                    total_swap_out_count += out_cnt
                    total_swap_latency   += out_lat

        # -------------------------------------------------------
        # 5) 对比图
        # -------------------------------------------------------
        if len(request_lengths) > 0:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(request_lengths, moe_latencies, label='MoE Latency', marker='o')
                plt.plot(request_lengths, total_times, label='Total Inference Time', marker='^')
                plt.xlabel("Request Length (tokens)")
                plt.ylabel("Runtime (seconds)")
                plt.title("Expert-level Swap (Dynamic used_experts detection)")
                plt.legend()
                plt.grid(True)
                plt.savefig("moe_swap_expert_weights_only.png")
                plt.show()
            except Exception as e:
                print(f"[main] 绘图时异常: {e}")
                traceback.print_exc()

        print(f"[main] 总计: swap_in={total_swap_in_count}, swap_out={total_swap_out_count}, swap_latency={total_swap_latency:.4f}s")

    except Exception as e:
        print(f"[main] 请求处理阶段出错: {e}")
        traceback.print_exc()

if __name__=="__main__":
    torch.cuda.empty_cache()
    main()
