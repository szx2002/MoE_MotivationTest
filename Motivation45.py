import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from safetensors import safe_open
import matplotlib.pyplot as plt
import random

def main():
    token = "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"
    login(token)
    
    # 显存限制为10GB
    max_memory = {0: "10GB"}
    
    # 可选4-bit量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type="fp16"
    )

    print("开始加载配置与分词器...")
    # 加载模型配置（不自动加载权重）
    config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1", token=token)
    # 基于config初始化空模型
    model = AutoModelForCausalLM.from_config(config)
    model.to("cpu")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1",
        use_fast=True,
        token=token
    )

    # 读取map文件
    map_file = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1/model.safetensors.index.json"
    with open(map_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    # 只加载前五个分片文件
    allowed_shards = [f"model-{i:05d}-of-00019.safetensors" for i in range(1, 6)]
    filtered_map = {p: s for p, s in weight_map.items() if s in allowed_shards}

    shard_dict = {}
    model_dir = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
    for param_name, shard_file in filtered_map.items():
        if shard_file not in shard_dict:
            shard_dict[shard_file] = []
        shard_dict[shard_file].append(param_name)

    # 加载这5个文件中的参数
    for shard_file, params in shard_dict.items():
        shard_path = os.path.join(model_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for p_name in params:
                tensor = f.get_tensor(p_name)
                sub_module = model
                sub_names = p_name.split(".")
                param_name = sub_names[-1]
                module_path = sub_names[:-1]

                for name_part in module_path:
                    if hasattr(sub_module, name_part):
                        sub_module = getattr(sub_module, name_part)
                    else:
                        # 没有对应的子模块则跳过
                        sub_module = None
                        break
                if sub_module is not None:
                    if param_name in sub_module._parameters:
                        tensor = tensor.to(torch.float16)
                        sub_module._parameters[param_name] = torch.nn.Parameter(tensor)
                    elif param_name in sub_module._buffers:
                        sub_module._buffers[param_name] = tensor

    model = model.to("cuda")
    print("已仅加载前五个model文件中的参数。（模型不完整）")

    # 准备请求
    input_file = "requests.txt"
    requests = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            req = line.strip()
            if req:
                requests.append(req)

    # 获取 num_experts 和 num_layers
    num_experts = 0
    num_layers = 0
    if len(requests) > 0:
        test_req = requests[0]
        test_inputs = tokenizer(test_req, return_tensors="pt").to("cuda")
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

    # 模拟expert管理与swap逻辑
    max_experts_in_gpu = 10
    experts_in_gpu = set()   # (layer_idx, expert_id)

    # 构建expert到文件的映射(概念性代码)
    expert_to_files = {}
    for full_key, shard_file in index_data["weight_map"].items():
        parts = full_key.split(".")
        if "experts" in parts:
            layer_idx = parts[2]
            expert_id = parts[5]
            expert_key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            if expert_key_prefix not in expert_to_files:
                expert_to_files[expert_key_prefix] = []
            expert_to_files[expert_key_prefix].append((full_key, shard_file))

    def load_expert_weights(layer_idx, expert_id):
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
        if prefix not in expert_to_files:
            return 0.0
        start_t = time.perf_counter()
        # 仅加载在allowed_shards中的参数
        shard_groups = {}
        for p_name, s_file in expert_to_files[prefix]:
            if s_file in allowed_shards:
                if s_file not in shard_groups:
                    shard_groups[s_file] = []
                shard_groups[s_file].append(p_name)

        for shard_file, params in shard_groups.items():
            shard_path = os.path.join(model_dir, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for p_name in params:
                    tensor = f.get_tensor(p_name)
                    # 实际需要写入expert参数，此处仅示意
                    # model.set_expert_param(layer_idx, expert_id, p_name, tensor.to(torch.float16))
                    pass
        # model.to_expert_cuda(layer_idx, expert_id)
        end_t = time.perf_counter()
        return end_t - start_t

    total_swap_in_count = 0
    total_swap_out_count = 0
    total_swap_latency = 0.0

    request_lengths = []
    total_times = []
    moe_latencies = []

    for req_idx, req in enumerate(requests):
        input_ids = tokenizer(req, return_tensors="pt").input_ids
        req_length = input_ids.shape[1]
        request_lengths.append(req_length)

        inputs = tokenizer(req, return_tensors="pt").to("cuda")

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

        moe_latency = outputs['latency']
        moe_latencies.append(moe_latency)
        total_times.append(total_runtime)

        print(f"请求 {req_idx+1} 完成：请求长度 {req_length}")
        print(f"  MoE层运行时间(来自outputs['latency']): {moe_latency:.4f}s")
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
        print("router_logits 是一个 tuple，长度为:", len(router_logits_tuple))

        if len(router_logits_tuple) == 0:
            print("没有router logits输出。")
            continue

        example_shape = router_logits_tuple[0].shape
        print(f"每个 router_logits 元素的形状: {example_shape}")

        selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]
        selected_experts = torch.stack(selected_experts, dim=-1)
        print("selected_experts shape:", selected_experts.shape)

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
                    gpu_experts_list = list(experts_in_gpu)
                    random.shuffle(gpu_experts_list)
                    expert_to_remove = None
                    for candidate in gpu_experts_list:
                        if candidate not in used_experts:
                            expert_to_remove = candidate
                            break
                    if expert_to_remove is None:
                        expert_to_remove = gpu_experts_list[0]

                    start_t = time.perf_counter()
                    # model.to_expert_cpu(*expert_to_remove)
                    experts_in_gpu.remove(expert_to_remove)
                    end_t = time.perf_counter()
                    request_swap_out_count += 1
                    request_swap_latency += (end_t - start_t)

                load_time = load_expert_weights(l_idx, e_id)
                experts_in_gpu.add((l_idx, e_id))
                request_swap_in_count += 1
                request_swap_latency += load_time

        total_swap_in_count += request_swap_in_count
        total_swap_out_count += request_swap_out_count
        total_swap_latency += request_swap_latency

        print(f"本请求swap统计：swap in次数={request_swap_in_count}, swap out次数={request_swap_out_count}, swap操作总延时={request_swap_latency:.4f}s\n")

    # 绘制对比图：MoE latency vs Total Runtime
    if len(request_lengths) > 0:
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

    print(f"所有请求合计：swap in次数={total_swap_in_count}, swap out次数={total_swap_out_count}, swap操作总延时={total_swap_latency:.4f}s")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
