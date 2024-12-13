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
        
        # 假设 MoE 层类名或命名中包含 "MoE" 字样，这里做一个简单判断示例：
        # 请根据实际模型结构修改判断条件。
        def is_moe_layer(layer_name, layer_module):
            # 假设当层名中有"moe"或类名中有"MoE"即为MoE层
            # 实际需根据模型架构进行精确判断
            name_lower = layer_name.lower()
            class_name_lower = layer_module.__class__.__name__.lower()
            if "moe" in name_lower or "moe" in class_name_lower:
                return True
            return False

        # 全局字典，用于存储各类延迟
        # 我们将所有 MoE 层的时间和所有普通层的时间分别累计
        moe_layer_time = 0.0
        normal_layer_time = 0.0

        # 为了记录每一层的开始时间，需要在forward前后hook中做记录
        layer_start_times = {}

        def forward_pre_hook(layer, input):
            layer_start_times[id(layer)] = time.time()

        def forward_hook(layer, input, output):
            elapsed = time.time() - layer_start_times[id(layer)]
            if is_moe_layer(layer_name_map[layer], layer):
                nonlocal moe_layer_time
                moe_layer_time += elapsed
            else:
                nonlocal normal_layer_time
                normal_layer_time += elapsed

        # 注册hook
        # 为了在hook中识别层，我们需要事先存储 name->module 的映射
        layer_name_map = {}
        for name, module in model.model.named_modules():
            # 判断是否是transformer层，通常transformer层在 model.model.layers 下
            if "model.layers." in name and (len(list(module.children())) == 0):
                # 叶子模块作为一个层的代表
                layer_name_map[module] = name
                module.register_forward_pre_hook(forward_pre_hook)
                module.register_forward_hook(forward_hook)

        # 准备不同长度的请求输入，用于测试延迟随长度的变化
        # 您可以根据需要调整不同的请求长度范围
        prompt_lengths = [8, 16, 32, 64, 128, 256, 512]
        
        moe_times = []
        normal_times = []

        # 使用一个基础的prompt句子
        base_prompt = "What is artificial intelligence? "
        
        for pl in prompt_lengths:
            prompt = base_prompt + ("A" * pl)  # 通过增加"A"的数量来增加输入长度
            
            # 清空计时统计
            moe_layer_time = 0.0
            normal_layer_time = 0.0

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                # 仅生成少量token，以减少生成过程干扰，这里设置max_new_tokens为1
                # 我们的关注点是forward对输入的处理延迟
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
            
            # 记录本长度下的统计结果
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
