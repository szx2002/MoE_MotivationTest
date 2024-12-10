from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import warnings
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
            "C:\\Uconn\\MoETest\\huggingfaceM87Bv01",
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
        
        # 测试生成
        prompt = "What is artificial intelligence? Answer in one sentence."
        print(f"\n输入提示: {prompt}")
        
        # 生成文本
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000000,     # 减少生成长度
                num_return_sequences=1,
                do_sample=True,        # 启用采样
                temperature=0.7,       # 调整温度
                top_p=0.95,           # 使用 top_p 采样
                repetition_penalty=1.1 # 避免重复
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n生成结果: {generated_text}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 清理 GPU 缓存
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