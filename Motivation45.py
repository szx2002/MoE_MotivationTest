import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from accelerate import load_checkpoint_and_dispatch
import matplotlib.pyplot as plt
import random
import traceback

def main():
    token = "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"
    login(token)
    
    print("开始加载模型...")
    try:
        # 使用4-bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
            # 移除 'bnb_4bit_compute_type'
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1",
            use_auth_token=token  # 使用 'use_auth_token' 而不是 'token'
        )
        
        # 设置设备映射
        max_memory = {
            "cpu": "16GB",
            "cuda:0": "10GB"
        }

        device_map = {}
    
        # 分配 embed_tokens 到 GPU0
        device_map["model.embed_tokens"] = "cuda:0"
    
        # 分配 layers.0 至 layers.4 到 GPU0
        for i in range(0, 5):
            layer_name = f"model.layers.{i}"
            device_map[layer_name] = "cuda:0"
    
        # 分配 layers.5 至 layers.31 到 CPU
        for i in range(5, 32):
            layer_name = f"model.layers.{i}"
            device_map[layer_name] = "cpu"
    
        # 分配 norm 和 lm_head 到 CPU
        device_map["model.norm"] = "cpu"
        device_map["lm_head"] = "cpu"
    
        # 加载模型并应用 device_map
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            use_auth_token=token,  # 使用 'use_auth_token' 而不是 'token'
            torch_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,  # 启用 FP32 CPU offload
            trust_remote_code=True,  # 如果模型使用自定义代码，需设置为 True
            use_safetensors=True      # 使用 safetensors 格式
        )
        
        # 打印模型的设备分配情况
        for layer, device in model.hf_device_map.items():
            print(f"{layer}: {device}")
        
        print("模型加载完成！")
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return  # 终止程序，避免后续代码出错
    
    print("模型加载完成，开始处理请求...")
    
    try:
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
            test_inputs = tokenizer(test_req, return_tensors="pt").to("cuda:0")  # 确保输入在正确的设备上
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
        # 此部分需要根据具体模型结构调整
        # 这里假设模型中有block_sparse_moe.experts结构
        # 您可能需要根据实际情况调整以下代码
        expert_to_files = {}
        model_dir = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
        # 遍历所有参数，构建专家映射
        for name, param in model.named_parameters():
            if "block_sparse_moe.experts" in name:
                parts = name.split(".")
                layer_idx = parts[2]
                expert_id = parts[5]
                expert_key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
                if expert_key_prefix not in expert_to_files:
                    expert_to_files[expert_key_prefix] = []
                # 这里假设每个专家对应一个文件，实际情况需要根据模型结构调整
                # 例如，如果每个专家参数分布在多个文件中，需要进一步处理
                expert_to_files[expert_key_prefix].append(name)
    
        def load_expert_weights(layer_idx, expert_id):
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            if prefix not in expert_to_files:
                return 0.0
            start_t = time.perf_counter()
            # 由于我们使用了 device_map，这里可以简化加载逻辑
            # 具体实现取决于模型的具体结构和需求
            # 这里只是一个示例
            # 您可能需要根据实际情况调整
            # 例如，重新分配设备，或触发某些加载机制
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
    
            inputs = tokenizer(req, return_tensors="pt").to("cuda:0")  # 确保输入在正确的设备上
    
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
    
            print(f"请求 {req_idx+1} 完成：请求长度 {req_length}")
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
                        # 由于我们使用了 device_map，这里可以简化移除逻辑
                        # 具体实现取决于模型的具体结构和需求
                        # 这里只是一个示例
                        # 您可能需要根据实际情况调整
                        # 例如，重新分配设备，或触发某些卸载机制
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
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
