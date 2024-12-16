import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import warnings
from collections import defaultdict

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
        
        # 出现的Expert统计
        activated_experts_per_layer = defaultdict(set)
        
        def is_moe_layer(name, module):
            # 修改匹配规则，确保识别 MoE 层
            return "block_sparse_moe" in name and "experts" in name
        
        def expert_hook(layer_name):
            def hook(module, inp, out):
                for expert_name, expert_module in module.named_modules():
                    if "experts" in expert_name:
                        expert_idx = int(expert_name.split(".")[-3])
                        activated_experts_per_layer[layer_name].add(expert_idx)
            return hook
        
        def register_hooks_for_submodules(parent_module, parent_name=""):
            for name, sub_module in parent_module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if is_moe_layer(full_name, sub_module):
                    sub_module.register_forward_hook(expert_hook(full_name))
                register_hooks_for_submodules(sub_module, full_name)
        
        # 注册 Hook
        register_hooks_for_submodules(model)

        input_file = "requests.txt"
        requests = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                req = line.strip()
                if req:
                    requests.append(req)
        
        for req_idx, req in enumerate(requests):
            activated_experts_per_layer.clear()
            inputs = tokenizer(req, return_tensors="pt").to("cuda")
            
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False
                )
            
            total_activated_experts = sum(len(experts) for experts in activated_experts_per_layer.values())
            print(f"\n请求 {req_idx + 1}:")
            for layer_name, experts in sorted(activated_experts_per_layer.items()):
                print(f"  {layer_name}: {len(experts)} 个 Experts 激活")
            print(f"  激活的 Experts 总数: {total_activated_experts}")
        
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
