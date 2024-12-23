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
import traceback  # 确保在顶部导入

def move_expert_to_device(model, layer_idx, expert_idx, device):
    """
    将指定层和专家的所有参数及缓冲区移动到目标设备（CPU 或 GPU）。
    """
    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
    try:
        expert = model.get_submodule(prefix)
        expert.to(device)
        print(f"已将 {prefix} 移动到 {device}")
        return True
    except AttributeError as e:
        print(f"模型中未找到 {prefix}: {e}")
        return False

def swap_in_expert(model, layer_idx, expert_idx, experts_in_gpu):
    """
    将指定专家移动到 GPU。
    """
    start_t = time.perf_counter()
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cuda")
    end_t = time.perf_counter()
    if moved:
        experts_in_gpu.add((layer_idx, expert_idx))
        swap_latency = end_t - start_t
        return 1, 0, swap_latency
    else:
        return 0, 0, 0.0

def swap_out_expert(model, layer_idx, expert_idx, experts_in_gpu):
    """
    将指定专家移动到 CPU。
    """
    start_t = time.perf_counter()
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cpu")
    end_t = time.perf_counter()
    if moved:
        experts_in_gpu.discard((layer_idx, expert_idx))
        swap_latency = end_t - start_t
        return 0, 1, swap_latency
    else:
        return 0, 0, 0.0

def main():
    try:
        print("开始加载模型...")

        # 检查GPU可用性和显存
        print("\n检查GPU状态...")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 显存: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)} GB")
        else:
            print("没有可用的 GPU。")
            return

        # 使用环境变量管理 token，提升安全性（推荐）
        token = os.getenv("HUGGINGFACE_TOKEN", "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")  # 请替换为您的实际 token 或设置环境变量
        if not token:
            print("请设置环境变量 HUGGINGFACE_TOKEN")
            return
        login(token)

        # 使用4-bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
            # 移除 'bnb_4bit_compute_type'
        )

        # 加载模型并将其移到 GPU
        model_id = "mistralai/Mixtral-8x7B-v0.1"  # 请替换为您的实际模型 ID
        print(f"加载模型 {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            token=token,  # 使用 'token' 而不是 'use_auth_token'
            torch_dtype=torch.float16,
            trust_remote_code=True,  # 如果模型使用自定义代码，需设置为 True
            use_safetensors=True      # 使用 safetensors 格式
        )

        print("将整个模型移到 GPU...")
        model.to("cuda")

        # 加载分词器
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token  # 使用 'token' 而不是 'use_auth_token'
        )

        # 打印模型的设备分配情况
        print("\n模型加载后设备分配:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")

        # 打印所有模块名称
        print("\n模型的所有模块名称:")
        for name, module in model.named_modules():
            print(name)

        model.eval()
        print("模型加载完成！")

    except Exception as e:
        print(f"错误在模型加载阶段: {str(e)}")
        traceback.print_exc()
        return  # 终止程序，避免后续代码出错

    print("\n模型加载完成，开始处理请求...")

    try:
        # 准备请求
        input_file = "requests.txt"
        requests = []
        if os.path.exists(input_file):
            print(f"加载请求文件 '{input_file}'...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    req = line.strip()
                    if req:
                        requests.append(req)
        else:
            print(f"请求文件 '{input_file}' 不存在。")
            return

        if not requests:
            print("没有有效的请求。")
            return

        # 获取 num_experts 和 num_layers
        print("\n获取模型的专家数量和层数...")
        num_experts = 0
        num_layers = 0
        test_req = requests[0]
        test_inputs = tokenizer(test_req, return_tensors="pt").to("cuda")  # 确保输入在 GPU 上
        with torch.inference_mode():
            test_outputs = model(
                **test_inputs,
                output_router_logits=True,
                return_dict=True,
                use_cache=False
            )
        router_logits_test = test_outputs.router_logits
        if len(router_logits_test) > 0:
            num_experts = router_logits_test[0].shape[1]
            num_layers = len(router_logits_test)

        print(f"模型中专家数量: {num_experts}, 层数: {num_layers}")

        # 模拟expert管理与swap逻辑
        max_experts_in_gpu = 20  # 调整GPU内存映射至20GB
        experts_in_gpu = set()   # (layer_idx, expert_id)

        # 构建expert到文件的映射(概念性代码)
        # 此部分需要根据具体模型结构调整
        expert_to_files = {}
        model_dir = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
        # 遍历所有参数，构建专家映射
        for name, param in model.named_parameters():
            if "block_sparse_moe.experts" in name:
                parts = name.split(".")
                layer_idx = int(parts[2])
                expert_id = int(parts[5])
                expert_key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
                if expert_key_prefix not in expert_to_files:
                    expert_to_files[expert_key_prefix] = []
                expert_to_files[expert_key_prefix].append(name)

        # 在此阶段不移动任何专家到 CPU，确保首次前向传播时所有必要的专家在 GPU 上
        # 如果需要，可以在首次前向传播之后移动部分专家到 CPU

        total_swap_in_count = 0
        total_swap_out_count = 0
        total_swap_latency = 0.0

        request_lengths = []
        total_times = []
        moe_latencies = []

        for req_idx, req in enumerate(requests):
            print(f"\n处理请求 {req_idx+1}/{len(requests)}: '{req}'")
            input_ids = tokenizer(req, return_tensors="pt").input_ids
            req_length = input_ids.shape[1]
            request_lengths.append(req_length)

            inputs = tokenizer(req, return_tensors="pt").to("cuda")  # 确保输入在 GPU 上

            # 使用CUDA事件计时总时间
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

            # 假设outputs包含latency信息，这需要根据实际模型输出调整
            # 如果没有，可以将其替换为其他合适的指标
            # 这里假设latency为total_runtime的一部分
            moe_latency = total_runtime * 0.5  # 示例值
            moe_latencies.append(moe_latency)
            total_times.append(total_runtime)

            print(f"  MoE层运行时间(假设值): {moe_latency:.4f}s")
            print(f"  整个预测过程运行时间(使用CUDA事件计时): {total_runtime:.4f}s")

            # 获取router_logits进行expert统计
            with torch.inference_mode():
                router_outputs = model(
                    **inputs,
                    output_router_logits=True,
                    return_dict=True,
                    use_cache=False
                )

            router_logits_tuple = router_outputs.router_logits
            print(f"  router_logits 是一个 tuple，长度为: {len(router_logits_tuple)}")

            if len(router_logits_tuple) == 0:
                print("  没有router logits输出。")
                continue

            example_shape = router_logits_tuple[0].shape
            print(f"  每个 router_logits 元素的形状: {example_shape}")

            selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]
            selected_experts = torch.stack(selected_experts, dim=-1)
            print(f"  selected_experts shape: {selected_experts.shape}")

            sequence_length, num_layers_actual = selected_experts.shape
            num_total_experts = 0

            used_experts = []
            for layer_idx in range(num_layers_actual):
                experts_in_layer = selected_experts[:, layer_idx].tolist()
                unique_experts_in_layer = set(experts_in_layer)
                num_unique_experts = len(unique_experts_in_layer)
                num_total_experts += num_unique_experts
                print(f"  Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}")
                for e_id in unique_experts_in_layer:
                    used_experts.append((layer_idx, e_id))

            print(f"  该请求中激活的专家总数 = {num_total_experts}")

            used_experts = list(set(used_experts))

            # Expert Swap in/out逻辑
            request_swap_in_count = 0
            request_swap_out_count = 0
            request_swap_latency = 0.0

            for (l_idx, e_id) in used_experts:
                if (l_idx, e_id) not in experts_in_gpu:
                    if len(experts_in_gpu) >= max_experts_in_gpu:
                        # 选择一个随机的专家进行swap out
                        expert_to_remove = random.choice(list(experts_in_gpu))
                        print(f"  GPU已满，正在将专家 {expert_to_remove} 移动回 CPU")
                        swap_out, _, latency_out = swap_out_expert(model, expert_to_remove[0], expert_to_remove[1], experts_in_gpu)
                        request_swap_out_count += swap_out
                        request_swap_latency += latency_out

                    # Swap in the required expert
                    print(f"  将专家 ({l_idx}, {e_id}) 移动到 GPU")
                    swap_in, _, latency_in = swap_in_expert(model, l_idx, e_id, experts_in_gpu)
                    request_swap_in_count += swap_in
                    request_swap_latency += latency_in

            total_swap_in_count += request_swap_in_count
            total_swap_out_count += request_swap_out_count
            total_swap_latency += request_swap_latency

            print(f"  本请求swap统计：swap in次数={request_swap_in_count}, swap out次数={request_swap_out_count}, swap操作总延时={request_swap_latency:.4f}s\n")

    except Exception as e:
        print(f"错误在请求处理阶段: {str(e)}")
        traceback.print_exc()
        return  # 终止程序，避免后续代码出错

    try:
        # 绘制对比图：MoE latency vs Total Runtime
        if len(request_lengths) > 0:
            print("\n绘制对比图：MoE latency vs Total Runtime...")
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
        print(f"错误在绘图阶段: {str(e)}")
        traceback.print_exc()

    print(f"\n所有请求合计：swap in次数={total_swap_in_count}, swap out次数={total_swap_out_count}, swap操作总延时={total_swap_latency:.4f}s")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
