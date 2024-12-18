import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import warnings
import matplotlib.pyplot as plt
import random
from safetensors import safe_open  # 假设使用safetensors库进行部分参数加载

warnings.filterwarnings('ignore')

def main():
    token = "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"
    login(token)
    
    # 原先为20GB, 现在改为10GB
    max_memory = {0: "10GB"}
    
    print("开始加载模型...")
    try:
        # 使用4-bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="fp16"
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1",
            use_fast=True,
            token=token
        )
        
        device_map = {
            "model.embed_tokens": 0,
            "model.layers": 0,
            "model.norm": 0,
            "lm_head": 0
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            token=token,
            torch_dtype=torch.float16
        )
        
        print("模型加载完成！")

        # 读取map文件
        map_file = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1/model.safetensors.index.json"
        with open(map_file, "r") as f:
            index_data = json.load(f)
        
        # index_data 的结构通常是：
        # {
        #   "metadata": {...},
        #   "weight_map": {
        #       "model.layers.0.block_sparse_moe.experts.0.w1.weight": "model-00001-of-00019.safetensors",
        #       ...
        #   }
        # }

        # 我们需要根据expert来组织这些数据，比如 expert: {param_name: file_name}
        # expert_name参考："model.layers.{l}.block_sparse_moe.experts.{e_id}"
        # 我们把上述字符串作为expert的key前缀。
        
        expert_to_files = {}
        for full_key, shard_file in index_data["weight_map"].items():
            # full_key例子: "model.layers.0.block_sparse_moe.experts.0.w1.weight"
            # 我们要解析layer和expert id
            # 假设expert key形式固定："model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}."
            # 我们可先split:
            parts = full_key.split(".")
            # parts = ["model", "layers", "{layer_id}", "block_sparse_moe", "experts", "{expert_id}", "w1", "weight"]
            if "experts" in parts:
                layer_idx = parts[2]
                expert_id = parts[5]
                # expert key前缀
                expert_key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"

                if expert_key_prefix not in expert_to_files:
                    expert_to_files[expert_key_prefix] = []
                expert_to_files[expert_key_prefix].append((full_key, shard_file))

        # 现在expert_to_files中存储了每个expert对应的(参数名, 分片文件)列表

        # 准备请求输入
        input_file = "requests.txt"
        requests = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                req = line.strip()
                if req:
                    requests.append(req)
        
        request_lengths = []
        total_times = []
        moe_latencies = []  # 从outputs['latency']获取MoE latency
        
        # swap相关计数和延迟
        total_swap_in_count = 0
        total_swap_out_count = 0
        total_swap_latency = 0.0

        # 假设一次前向提取router logits信息，以获取num_experts和num_layers
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
            num_experts = router_logits_test[0].shape[1]
            num_layers = len(router_logits_test)
        else:
            num_experts = 0
            num_layers = 0

        # 模拟最大在GPU同时驻留的expert数量（用数量限制来模拟显存不足）
        max_experts_in_gpu = 10
        experts_in_gpu = set()   # (layer_idx, expert_id) 在GPU中的expert集合

        def load_expert_weights(layer_idx, expert_id):
            # 根据expert prefix找到相应权重
            prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
            if prefix not in expert_to_files:
                # 说明这个expert在map中不存在
                return
            
            # 开始计时
            start_t = time.perf_counter()
            # 假设从对应的safetensor分片加载参数到model对应的位置
            # 在实际实现中，需要知道expert结构在model中对应的nn.Module并将参数赋值。
            # 这里仅作概念演示：
            
            # 我们可以循环所有 (param_name, shard_file)
            shard_groups = {}
            for param_name, shard_file in expert_to_files[prefix]:
                if shard_file not in shard_groups:
                    shard_groups[shard_file] = []
                shard_groups[shard_file].append(param_name)
            
            # 打开对应的safetensor文件并加载参数
            # 注意：实际中需要先知道expert的参数形状和结构，本例中仅演示流程
            for shard_file, params in shard_groups.items():
                shard_path = os.path.join("/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1", shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for p_name in params:
                        tensor = f.get_tensor(p_name)
                        # 假设我们能通过 model指针拿到对应expert的参数并赋值
                        # 需要在模型中有访问专家模块的方式，这里不实现具体细节
                        # 例如: model.set_expert_param(layer_idx, expert_id, p_name, tensor)
                        # 此处仅作假设
                        pass

            # 将expert的参数迁移到GPU
            # 这里同样假设有方法访问并to("cuda")
            # model.to_expert_cuda(layer_idx, expert_id)
            
            end_t = time.perf_counter()
            return end_t - start_t

        # 对每个请求进行处理
        for req_idx, req in enumerate(requests):
            input_ids = tokenizer(req, return_tensors="pt").input_ids
            req_length = input_ids.shape[1]  # token数
            request_lengths.append(req_length)

            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            # 使用model.generate进行推理
            # 假设outputs中存在 'latency' 字段来表示MoE层latency（需模型支持）
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False,
                    output_latency=True  # 假设模型支持此参数返回 {'latency': float}
                )
            # 假设outputs是一个dict或有属性存储latency信息
            # 如果是Hugging Face标准generate返回值，一般是tensor，但这里假定已扩展支持
            moe_latency = outputs['latency']
            total_runtime = outputs['total_time'] if 'total_time' in outputs else 0.0
            if total_runtime == 0.0:
                # 如果没有提供total_time，就不进行total_times记录
                # 或使用CUDA事件计时也可，但用户已明确要求不使用之前的hook逻辑
                total_runtime = moe_latency  # 临时假设

            moe_latencies.append(moe_latency)
            total_times.append(total_runtime)

            print(f"请求 {req_idx+1} 完成：请求长度 {req_length}")
            print(f"  MoE层运行时间(来自outputs['latency']): {moe_latency:.4f}s")
            print(f"  整个预测过程运行时间: {total_runtime:.4f}s")

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
                print("没有路由器 logits 输出。")
                continue
            
            example_shape = router_logits_tuple[0].shape
            print(f"每个 router_logits 元素的形状: {example_shape}")
            
            selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]
            selected_experts = torch.stack(selected_experts, dim=-1)  # [sequence_length, num_layers]
            print("selected_experts shape:", selected_experts.shape)
            
            sequence_length, num_layers_actual = selected_experts.shape
            total_activated_experts = set()
            num_total_experts = 0

            for layer_idx in range(num_layers_actual):
                experts_in_layer = selected_experts[:, layer_idx].tolist()
                unique_experts_in_layer = set(experts_in_layer)
                num_unique_experts = len(unique_experts_in_layer)
                num_total_experts += num_unique_experts
                print(f"  Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}")
                total_activated_experts.update(unique_experts_in_layer)
            
            print(f"  该请求中激活的专家总数 = {num_total_experts}")

            # Expert Swap In/Out逻辑
            request_swap_in_count = 0
            request_swap_out_count = 0
            request_swap_latency = 0.0

            used_experts = []
            for layer_idx in range(num_layers_actual):
                experts_in_layer = selected_experts[:, layer_idx].tolist()
                unique_exp = set(experts_in_layer)
                for e_id in unique_exp:
                    used_experts.append((layer_idx, e_id))

            used_experts = list(set(used_experts))

            # 检查并加载需要的expert
            for (l_idx, e_id) in used_experts:
                expert_key_prefix = f"model.layers.{l_idx}.block_sparse_moe.experts.{e_id}"
                # 如果expert不在GPU，需要swap in
                if (l_idx, e_id) not in experts_in_gpu:
                    # GPU已满则swap out一个不在used_experts中的expert
                    if len(experts_in_gpu) >= max_experts_in_gpu:
                        gpu_experts_list = list(experts_in_gpu)
                        random.shuffle(gpu_experts_list)
                        expert_to_remove = None
                        for candidate in gpu_experts_list:
                            if candidate not in used_experts:
                                expert_to_remove = candidate
                                break
                        if expert_to_remove is None:
                            # 如果全在用，只能移除一个，但这会影响性能
                            expert_to_remove = gpu_experts_list[0]
                        
                        # swap out
                        start_t = time.perf_counter()
                        # 将expert_to_remove移回CPU或释放GPU资源
                        # 实际上需要调用model接口将对应专家权重to("cpu")
                        # model.to_expert_cpu(*expert_to_remove)
                        experts_in_gpu.remove(expert_to_remove)
                        end_t = time.perf_counter()
                        request_swap_out_count += 1
                        request_swap_latency += (end_t - start_t)

                    # swap in
                    load_time = load_expert_weights(l_idx, e_id)
                    if load_time is None:
                        load_time = 0.0
                    experts_in_gpu.add((l_idx, e_id))
                    request_swap_in_count += 1
                    request_swap_latency += load_time

            total_swap_in_count += request_swap_in_count
            total_swap_out_count += request_swap_out_count
            total_swap_latency += request_swap_latency

            print(f"本请求swap统计：swap in次数={request_swap_in_count}, swap out次数={request_swap_out_count}, swap操作总延时={request_swap_latency:.4f}s\n")

        # 绘制图表，MoE latency vs total time
        if len(request_lengths) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(request_lengths, moe_latencies, label='MoE Latency', marker='o')
            plt.plot(request_lengths, total_times, label='Total Inference Runtime', marker='^')
            plt.xlabel("Request Length (number of tokens)")
            plt.ylabel("Runtime (seconds)")
            plt.title("MoE Latency vs Total Inference Runtime")
            plt.legend()
            plt.grid(True)
            plt.savefig("moe_vs_total_runtime.png")
            plt.show()

        # 打印所有请求的swap统计结果
        print(f"所有请求合计：swap in次数={total_swap_in_count}, swap out次数={total_swap_out_count}, swap操作总延时={total_swap_latency:.4f}s")

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 版本: {bnb.__version__}")
    except:
        print("bitsandbytes 未安装或版本不正确")
        
    main()
