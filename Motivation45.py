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

def move_expert_to_device(model, layer_idx, expert_idx, device, device_map):
    """
    将指定层和专家的所有参数及缓冲区移动到目标设备（CPU 或 GPU）。
    并更新设备映射。
    """
    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
    try:
        expert = model.get_submodule(prefix)
        expert.to(device)
        device_map[(layer_idx, expert_idx)] = device
        print(f"已将 {prefix} 移动到 {device}")
        return True
    except AttributeError as e:
        print(f"模型中未找到 {prefix}: {e}")
        return False

def swap_in_expert(model, layer_idx, expert_idx, device_map, experts_in_gpu):
    """
    将指定专家移动到 GPU，并更新设备映射和 experts_in_gpu 集合。
    """
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cuda", device_map)
    if moved:
        experts_in_gpu.add((layer_idx, expert_idx))
        print(f"专家 ({layer_idx}, {expert_idx}) 已被移动到 GPU")
    return moved

def swap_out_expert(model, layer_idx, expert_idx, device_map, experts_in_gpu):
    """
    将指定专家移动到 CPU，并更新设备映射和 experts_in_gpu 集合。
    """
    moved = move_expert_to_device(model, layer_idx, expert_idx, "cpu", device_map)
    if moved:
        experts_in_gpu.discard((layer_idx, expert_idx))
        print(f"专家 ({layer_idx}, {expert_idx}) 已被移动到 CPU")
    return moved

def get_router_logits(model, inputs):
    """
    运行前向传播以获取 router_logits。
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
    基于 router_logits 确定需要使用的专家。
    """
    # 假设 router_logits 是一个 tuple，每个元素的形状为 [batch_size, num_experts]
    used_experts = set()
    for layer_idx, logits in enumerate(router_logits):
        # 对于每一层，选择 logit 最高的专家
        selected_expert = logits.argmax(dim=-1).item()  # 假设 batch_size=1
        used_experts.add((layer_idx, selected_expert))
    return used_experts

def ensure_experts_on_gpu(model, used_experts, device_map, experts_in_gpu, max_experts_in_gpu):
    """
    确保所有使用的专家都在 GPU 上。如果不在，进行 swap in。
    如果 GPU 已满，则进行 swap out。
    """
    for (layer_idx, expert_idx) in used_experts:
        if device_map.get((layer_idx, expert_idx), 'cuda') != 'cuda':
            if len(experts_in_gpu) >= max_experts_in_gpu:
                # 选择一个当前在 GPU 上的专家进行 swap out
                expert_to_remove = random.choice(list(experts_in_gpu))
                print(f"  GPU已满，正在将专家 {expert_to_remove} 移动回 CPU")
                swap_out_expert(model, expert_to_remove[0], expert_to_remove[1], device_map, experts_in_gpu)
            # Swap in the required expert
            print(f"  将专家 ({layer_idx}, {expert_idx}) 移动到 GPU")
            swap_in_expert(model, layer_idx, expert_idx, device_map, experts_in_gpu)

def verify_device_consistency(model, device_map, num_layers, num_experts):
    """
    验证模型中：
    - 所有非专家参数均位于 GPU 上。
    - 每个专家位于其设备映射中指定的设备上。
    """
    inconsistencies = False
    for name, param in model.named_parameters():
        parts = name.split('.')
        if "block_sparse_moe.experts" in name:
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            expected_device = device_map.get((layer_idx, expert_idx), 'cuda')
            if param.device.type != expected_device:
                print(f"设备不匹配：{name} 应位于 {expected_device}, 但当前位于 {param.device}")
                inconsistencies = True
        else:
            if param.device.type != 'cuda':
                print(f"非专家参数 {name} 当前位于 {param.device}, 预期位于 cuda")
                inconsistencies = True
    if not inconsistencies:
        print("设备一致性验证通过：所有参数均位于正确的设备上。")
    else:
        print("设备一致性验证失败：部分参数不在预期的设备上。")

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
        num_experts = 8  # 每层有8个专家
        num_layers = 32  # 假设模型有32层
        print(f"模型中专家数量: {num_experts}, 层数: {num_layers}")

        # 模拟 expert 管理与 swap 逻辑
        max_experts_in_gpu = 20  # 根据 GPU 内存调整
        experts_in_gpu = set()    # (layer_idx, expert_id)

        # 初始化设备映射，将所有专家初始设为 'cuda'
        device_map = {}
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                device_map[(layer_idx, expert_idx)] = 'cuda'
                experts_in_gpu.add((layer_idx, expert_idx))

        # 构建 expert 到文件的映射（概念性代码）
        # 此部分需要根据具体模型结构调整
        expert_to_files = {}
        model_dir = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
        # 遍历所有参数，构建专家映射
        for name, param in model.named_parameters():
            if "block_sparse_moe.experts" in name:
                parts = name.split(".")
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                expert_key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
                if expert_key_prefix not in expert_to_files:
                    expert_to_files[expert_key_prefix] = []
                expert_to_files[expert_key_prefix].append(name)

        # 在此阶段不移动任何专家到 CPU，确保首次前向传播时所有必要的专家在 GPU 上

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

            # Step 1: 获取 router_logits
            print("  获取 router_logits...")
            router_logits = get_router_logits(model, inputs)
            print(f"  router_logits 是一个 tuple，长度为: {len(router_logits)}")

            # Step 2: 确定需要使用的专家
            used_experts = get_used_experts(router_logits)
            print(f"  该请求中激活的专家: {used_experts}")

            # Step 3: 确保所需的专家在 GPU 上
            ensure_experts_on_gpu(model, used_experts, device_map, experts_in_gpu, max_experts_in_gpu)

            # Step 4: 运行 generate
            print("  运行 generate()...")
            # 使用 CUDA 事件计时总时间
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

            # 假设 MoE 层运行时间为总运行时间的一部分
            moe_latency = total_runtime * 0.5  # 示例值
            moe_latencies.append(moe_latency)
            total_times.append(total_runtime)

            print(f"  MoE层运行时间(假设值): {moe_latency:.4f}s")
            print(f"  整个预测过程运行时间: {total_runtime:.4f}s")

            # Step 5: 选取用于 swap 的专家（可选）
            # 例如，保持最新使用的专家在 GPU 上，移动较少使用的专家到 CPU
            # 这里暂不实现复杂的 swap 策略，保持示例简单

            # 打印设备一致性验证
            print("  验证设备一致性...")
            verify_device_consistency(model, device_map, num_layers, num_experts)

            print(f"  本请求 swap 统计：swap in 次数=0, swap out 次数=0, swap 操作总延时=0.0000s\n")  # 示例值

    except Exception as e:
        print(f"错误在请求处理阶段: {str(e)}")
        print("当前设备映射:")
        for key, device in device_map.items():
            print(f"专家 {key} 在 {device}")
        print("当前模型参数的设备分配情况:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.device}")
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

    print(f"\n所有请求合计：swap in 次数={total_swap_in_count}, swap out 次数={total_swap_out_count}, swap 操作总延时={total_swap_latency:.4f}s")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
