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
        
        for i, layer_module in enumerate(model.model.layers):
            print(f"Layer {i}, Class: {layer_module.__class__.__name__}")
            for sub_name, sub_module in layer_module.named_modules():
                print(f"  Sub-module name: {sub_name}, class: {sub_module.__class__.__name__}")


        # 放宽MoE层判断标准
        # 如果层名或类名中包含 "moe", "block_sparse_moe", 或 "experts" 则认为是moe层
        def is_moe(name, module):
            name_lower = name.lower()
            class_name_lower = module.__class__.__name__.lower()
            moe_keywords = ["moe", "block_sparse_moe", "experts"]
            return any(kw in name_lower for kw in moe_keywords) or any(kw in class_name_lower for kw in moe_keywords)
        
        total_transformer_runtime = 0.0
        normal_transformer_runtime = 0.0
        moe_transformer_runtime = 0.0
        layer_start_times = {}

        # 记录首次出现的moe层和普通层名称
        first_moe_layer_name = None
        first_normal_layer_name = None

        def make_pre_hook(module_id):
            def forward_pre_hook(module, inp):
                torch.cuda.synchronize()
                layer_start_times[module_id] = time.time()
            return forward_pre_hook

        def make_post_hook(module_id, module_is_moe, layer_name):
            def forward_hook(module, inp, out):
                torch.cuda.synchronize()
                elapsed = time.time() - layer_start_times[module_id]
                nonlocal total_transformer_runtime, normal_transformer_runtime, moe_transformer_runtime
                nonlocal first_moe_layer_name, first_normal_layer_name

                total_transformer_runtime += elapsed
                if module_is_moe:
                    moe_transformer_runtime += elapsed
                    # 如果还没有记录moe层名字，就记录这一次的
                    if first_moe_layer_name is None:
                        first_moe_layer_name = layer_name
                else:
                    normal_transformer_runtime += elapsed
                    # 如果还没有记录普通层名字，就记录这一次的
                    if first_normal_layer_name is None:
                        first_normal_layer_name = layer_name
            return forward_hook

        # 注册hook
        # 同时保留层的名称以便在forward_hook中打印
        for name, layer_module in model.model.layers.named_children():
            module_id = id(layer_module)
            module_is_moe = is_moe(name, layer_module)
            layer_module.register_forward_pre_hook(make_pre_hook(module_id))
            layer_module.register_forward_hook(make_post_hook(module_id, module_is_moe, name))
        
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
            input_ids = tokenizer(req, return_tensors="pt").input_ids
            req_length = input_ids.shape[1]  # token数
            request_lengths.append(req_length)

            # 重置计时与首层名称记录
            total_transformer_runtime = 0.0
            normal_transformer_runtime = 0.0
            moe_transformer_runtime = 0.0
            layer_start_times.clear()
            first_moe_layer_name = None
            first_normal_layer_name = None
            
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
                    use_cache=False
                )
            torch.cuda.synchronize()
            end_total = time.time()
            
            total_runtime = end_total - start_total
            
            moe_times.append(moe_transformer_runtime)
            normal_times.append(normal_transformer_runtime)
            total_times.append(total_runtime)

            print(f"请求长度 {req_length} 完成：")
            print(f"  第一个MoE层: {first_moe_layer_name if first_moe_layer_name else '无MoE层'}")
            print(f"  第一个普通层: {first_normal_layer_name if first_normal_layer_name else '无普通层'}")
            print(f"  MoE层运行时间   {moe_transformer_runtime:.4f}s")
            print(f"  普通层运行时间 {normal_transformer_runtime:.4f}s")
            print(f"  整个预测过程运行时间 {total_runtime:.4f}s")
        
        # 绘制图表1：MoE层、普通层、总预测过程运行时间随请求长度变化
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

        # 绘制图表2：MoE层运行时间在整个预测过程中的占比
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
    
    # 确保 bitsandbytes 正确安装
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 版本: {bnb.__version__}")
    except:
        print("bitsandbytes 未安装或版本不正确")
        
    main()
