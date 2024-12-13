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
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            token=token,
            torch_dtype=torch.float16
        )
        
        print("模型加载完成！")
        
        # 根据实际情况修改判断是否为MoE层的逻辑
        def is_moe_layer(layer_module):
            class_name_lower = layer_module.__class__.__name__.lower()
            # 如果是MoE层的Block名字中有 "moe"
            # 请根据实际模型结构修改此判断条件
            return "moe" in class_name_lower
        
        moe_layer_time = 0.0
        normal_layer_time = 0.0
        layer_start_times = {}
        
        def make_pre_hook(idx, module_is_moe):
            def forward_pre_hook(module, inp):
                torch.cuda.synchronize()
                layer_start_times[id(module)] = time.time()
            return forward_pre_hook

        def make_post_hook(idx, module_is_moe):
            def forward_hook(module, inp, out):
                torch.cuda.synchronize()
                elapsed = time.time() - layer_start_times[id(module)]
                nonlocal moe_layer_time, normal_layer_time
                if module_is_moe:
                    moe_layer_time += elapsed
                else:
                    normal_layer_time += elapsed
            return forward_hook

        # 对model.model.layers中的每个transformer层注册hook
        for i, layer_module in enumerate(model.model.layers):
            module_is_moe = is_moe_layer(layer_module)
            layer_module.register_forward_pre_hook(make_pre_hook(i, module_is_moe))
            layer_module.register_forward_hook(make_post_hook(i, module_is_moe))
        
        # 准备不同长度的请求输入
        prompt_lengths = [8, 16, 32, 64, 128, 256, 512]
        
        moe_times = []
        normal_times = []

        base_prompt = "What is artificial intelligence? "
        
        for pl in prompt_lengths:
            prompt = base_prompt + ("A" * pl)
            
            # 重置计时统计
            moe_layer_time = 0.0
            normal_layer_time = 0.0
            layer_start_times.clear()
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                # 禁用cache以确保每次完整计算
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
            
            moe_times.append(moe_layer_time)
            normal_times.append(normal_layer_time)
            print(f"请求长度 {pl} 完成：MoE总延迟 {moe_layer_time:.4f}s, 普通层总延迟 {normal_layer_time:.4f}s")

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(prompt_lengths, moe_times, label='MoE Layers Latency', marker='o')
        plt.plot(prompt_lengths, normal_times, label='Transformer Layers Latency', marker='s')
        plt.xlabel("Request Length")
        plt.ylabel("Latency (seconds)")
        plt.title("MoE layer vs Transformer layer latency as request length increases")
        plt.legend()
        plt.grid(True)
        plt.savefig("moe_vs_transformer_latency.png")
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
