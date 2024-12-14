from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
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
        
        # 放宽MoE层判断标准
        # 如果层名或类名中包含 "moe", "block_sparse_moe", 或 "experts" 则认为是moe层
        def is_moe(name, module):
            name_lower = name.lower()
            class_name_lower = module.__class__.__name__.lower()
            # 根据已知weight_map中的特征，如 "block_sparse_moe" 和 "experts"
            # 如果层的名称或类名中出现这些字样，则认定为MoE层
            moe_keywords = ["moe", "block_sparse_moe", "experts"]
            return any(kw in name_lower for kw in moe_keywords) or any(kw in class_name_lower for kw in moe_keywords)
        
        total_transformer_time = 0.0
        normal_transformer_time = 0.0
        layer_start_times = {}

        def make_pre_hook(module_id):
            def forward_pre_hook(module, inp):
                torch.cuda.synchronize()
                layer_start_times[module_id] = time.time()
            return forward_pre_hook

        def make_post_hook(module_id, module_is_moe):
            def forward_hook(module, inp, out):
                torch.cuda.synchronize()
                elapsed = time.time() - layer_start_times[module_id]
                nonlocal total_transformer_time, normal_transformer_time
                # 所有transformer层计入total_transformer_time
                total_transformer_time += elapsed
                # 非MoE层计入normal_transformer_time
                if not module_is_moe:
                    normal_transformer_time += elapsed
            return forward_hook

        # 对 model.model.layers 下的每个子模块(层)进行注册
        # 使用 named_children() 来同时获取 name 和 module
        for name, layer_module in model.model.layers.named_children():
            module_id = id(layer_module)
            module_is_moe = is_moe(name, layer_module)
            layer_module.register_forward_pre_hook(make_pre_hook(module_id))
            layer_module.register_forward_hook(make_post_hook(module_id, module_is_moe))
        
        # 从文件中读取请求
        input_file = "requests.txt"
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

        for req in requests:
            # 对请求分词后获取长度
            input_ids = tokenizer(req, return_tensors="pt").input_ids
            req_length = input_ids.shape[1]  # token数
            request_lengths.append(req_length)

            # 重置计时统计
            total_transformer_time = 0.0
            normal_transformer_time = 0.0
            layer_start_times.clear()
            
            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            # 计时整个预测过程
            torch.cuda.synchronize()
            start_total = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False  # 确保每次完整forward
                )
            torch.cuda.synchronize()
            end_total = time.time()
            
            total_latency = end_total - start_total
            
            # 通过差值计算moe层延迟
            moe_layer_time = total_transformer_time - normal_transformer_time
            
            moe_times.append(moe_layer_time)
            normal_times.append(normal_transformer_time)
            total_times.append(total_latency)

            print(f"请求长度 {req_length} 完成：")
            print(f"  MoE总延迟   {moe_layer_time:.4f}s (通过差值得出)")
            print(f"  普通层总延迟 {normal_transformer_time:.4f}s")
            print(f"  整个预测过程延迟 {total_latency:.4f}s")
        
        # 绘制图表1：MoE层latency，普通层latency和总latency随请求长度变化
        plt.figure(figsize=(10, 6))
        plt.plot(request_lengths, moe_times, label='MoE Layers Latency', marker='o')
        plt.plot(request_lengths, normal_times, label='Normal Transformer Layers Latency', marker='s')
        plt.plot(request_lengths, total_times, label='Total Inference Latency', marker='^')
        plt.xlabel("Request Length (number of tokens)")
        plt.ylabel("Latency (seconds)")
        plt.title("MoE vs Normal Transformer Layers vs Total Latency")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_normal_total_latency.png")
        plt.show()

        # 绘制图表2：MoE层latency在整个预测延迟中的占比
        ratio = [(m / t) * 100.0 if t > 0 else 0.0 for m, t in zip(moe_times, total_times)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(request_lengths, ratio, label='MoE Latency Percentage', marker='o')
        plt.xlabel("Request Length (number of tokens)")
        plt.ylabel("MoE Latency Percentage (%)")
        plt.title("MoE Latency Share in Total Inference Latency")
        plt.grid(True)
        plt.savefig("moe_latency_percentage.png")
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
    
    # 确保 bitsandbytes 正确安装
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 版本: {bnb.__version__}")
    except:
        print("bitsandbytes 未安装或版本不正确")
        
    main()
