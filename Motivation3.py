import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="fp16"
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1",
            use_fast=True,  # 确保使用快速分词器
            token=token
        )
        
        # 设置设备映射
        max_memory = {0: "20GB"}
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

        # 以下为第一份代码中的 MoE 层和普通层时延监控相关实现
        def is_moe_layer(name, module):
            name_lower = name.lower()
            moe_keywords = ["block_sparse_moe", "experts"]
            return any(kw in name_lower for kw in moe_keywords)
        
        total_transformer_runtime = 0.0
        normal_transformer_runtime = 0.0
        moe_transformer_runtime = 0.0
        
        first_moe_layer_name = None
        first_normal_layer_name = None
        
        layer_start_events = {}

        def make_pre_hook(module_id):
            def forward_pre_hook(module, inp):
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                layer_start_events[module_id] = start_event
            return forward_pre_hook

        def make_post_hook(module_id, module_is_moe, layer_name):
            def forward_hook(module, inp, out):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                end_event.synchronize()
                start_event = layer_start_events[module_id]
                elapsed_ms = start_event.elapsed_time(end_event)  # 毫秒
                elapsed = elapsed_ms / 1000.0  # 转换成秒
                
                nonlocal total_transformer_runtime, normal_transformer_runtime, moe_transformer_runtime
                nonlocal first_moe_layer_name, first_normal_layer_name

                total_transformer_runtime += elapsed
                if module_is_moe:
                    moe_transformer_runtime += elapsed
                    if first_moe_layer_name is None:
                        first_moe_layer_name = layer_name
                else:
                    normal_transformer_runtime += elapsed
                    if first_normal_layer_name is None:
                        first_normal_layer_name = layer_name
            return forward_hook

        def register_hooks_for_submodules(parent_module, parent_name=""):
            for name, sub_module in parent_module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                module_is_moe = is_moe_layer(full_name, sub_module)
                sub_module.register_forward_pre_hook(make_pre_hook(id(sub_module)))
                sub_module.register_forward_hook(make_post_hook(id(sub_module), module_is_moe, full_name))
                register_hooks_for_submodules(sub_module, full_name)

        for i, layer_module in enumerate(model.model.layers):
            layer_name = f"layer_{i}"
            register_hooks_for_submodules(layer_module, layer_name)

        input_file = "requestM3.txt"
        requests = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                req = line.strip()
                if req:
                    requests.append(req)
        
        request_lengths = []
        moe_times = []
        normal_times = []
        total_times = []

        # 打开输出文件，准备写入 moe_latency 和 total_latency
        output_file = "results_M3.txt"
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 开始处理每个请求
            for req_idx, req in enumerate(requests):
                input_ids = tokenizer(req, return_tensors="pt").input_ids
                req_length = input_ids.shape[1]  # token数
                request_lengths.append(req_length)

                # 重置计时器
                total_transformer_runtime = 0.0
                normal_transformer_runtime = 0.0
                moe_transformer_runtime = 0.0
                first_moe_layer_name = None
                first_normal_layer_name = None
                layer_start_events.clear()
                
                inputs = tokenizer(req, return_tensors="pt").to("cuda")

                # 使用CUDA事件计时整体预测时间
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
                    print(outputs)
                total_end.record()
                total_end.synchronize()

                total_runtime_ms = total_start.elapsed_time(total_end)
                total_runtime = total_runtime_ms / 1000.0

                # 确保 outputs 包含 'latency' 键
                if 'latency' in outputs:
                    moe_transformer_runtime = outputs['latency']
                else:
                    # 如果没有 'latency'，使用通过hook记录的moe_transformer_runtime
                    pass  # 保持现有值
                
                normal_times.append(normal_transformer_runtime)
                total_times.append(total_runtime)
                moe_times.append(moe_transformer_runtime)

                # 将 moe_latency 和 total_latency 写入文件
                f_out.write(f"{moe_transformer_runtime} {total_runtime}\n")

                print(f"请求长度 {req_length} 完成：")
                print(f"  第一个MoE层: {first_moe_layer_name if first_moe_layer_name else '无MoE层'}")
                print(f"  第一个普通层: {first_normal_layer_name if first_normal_layer_name else '无普通层'}")
                print(f"  MoE层运行时间   {moe_transformer_runtime:.4f}s")
                print(f"  普通层运行时间 {normal_transformer_runtime:.4f}s")
                print(f"  整个预测过程运行时间 {total_runtime:.4f}s")

                # ========== 以下为第二份代码中关于 router_logits 的提取与分析部分 ==========
                # 在完成时延统计后，对同一输入再进行一次forward以获取router_logits
                with torch.inference_mode():
                    router_outputs = model(
                        **inputs,
                        output_router_logits=True,  # 启用路由器 logits 输出
                        return_dict=True,
                        use_cache=False  # 防止累积状态
                    )
                
                router_logits_tuple = router_outputs.router_logits
                print("router_logits 是一个 tuple，长度为:", len(router_logits_tuple))
                
                if len(router_logits_tuple) == 0:
                    print("没有路由器 logits 输出，检查模型配置和输出选项。")
                    continue
                
                example_shape = router_logits_tuple[0].shape
                print(f"每个 router_logits 元素的形状: {example_shape}")
                
                # 对每层进行 argmax，获取激活的专家
                selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]  
                selected_experts = torch.stack(selected_experts, dim=-1)  # [sequence_length, num_layers]
                print("selected_experts shape:", selected_experts.shape)
                
                print(f"\n请求 {req_idx + 1}: {req}")
                print("每一层中激活的专家数量以及请求中激活的专家总数:")

                sequence_length, num_layers = selected_experts.shape
                
                total_activated_experts = set()
                num_total_experts = 0

                for layer_idx in range(num_layers):
                    experts_in_layer = selected_experts[:, layer_idx].tolist()
                    unique_experts_in_layer = set(experts_in_layer)
                    num_unique_experts = len(unique_experts_in_layer)
                    num_total_experts += num_unique_experts
                    print(f"  Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}")
                    total_activated_experts.update(unique_experts_in_layer)
                
                print(f"  该请求中激活的专家总数 = {num_total_experts}\n")
                # ========== 第二份代码的逻辑到此结束 ==========

        # 绘图部分（来自第一份代码）
        plt.figure(figsize=(10, 6))
        plt.plot(request_lengths, moe_times, label='MoE Layers Runtime', marker='o')
        plt.plot(request_lengths, normal_times, label='Normal Transformer Layers Runtime', marker='s')
        plt.plot(request_lengths, total_times, label='Total Inference Runtime', marker='^')
        plt.xlabel("Request Length (number of tokens)")
        plt.ylabel("Runtime (seconds)")
        plt.title("MoE vs Normal Transformer Layers vs Total Runtime")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_normal_total_runtime.png")
        plt.show()

        ratio = [(m / t) * 100.0 if t > 0 else 0.0 for m, t in zip(moe_times, total_times)]
        plt.figure(figsize=(10, 6))
        plt.plot(request_lengths, ratio, label='MoE Runtime Percentage', marker='o')
        plt.xlabel("Request Length (number of tokens)")
        plt.ylabel("MoE Runtime Percentage (%)")
        plt.title("MoE Runtime Share in Total Inference Runtime")
        plt.grid(True)
        plt.savefig("moe_runtime_percentage.png")
        plt.show()

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
