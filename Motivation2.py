import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        
        input_file = "requestM3.txt"
        requests = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                req = line.strip()
                if req:
                    requests.append(req)
        
        # 打开 activated_experts.txt 文件以写入模式
        with open("activated_experts.txt", 'w', encoding='utf-8') as output_file:
            for req_idx, req in enumerate(requests):
                inputs = tokenizer(req, return_tensors="pt").to("cuda")
                
                with torch.inference_mode():
                    outputs = model(
                        **inputs,
                        output_router_logits=True,  # 启用路由器 logits 输出
                        return_dict=True,
                        use_cache=False  # 防止累积状态
                    )
                
                # outputs.router_logits 是一个包含每层 router_logits 的元组
                router_logits_tuple = outputs.router_logits
                print("router_logits 是一个 tuple，长度为:", len(router_logits_tuple))
                
                # 确保 tuple 非空
                if len(router_logits_tuple) == 0:
                    print("没有路由器 logits 输出，检查模型配置和输出选项。")
                    # 写入 0 作为该请求的激活专家总数
                    output_file.write("0\n")
                    continue
                
                # 检查每个元素的形状
                example_shape = router_logits_tuple[0].shape
                print(f"每个 router_logits 元素的形状: {example_shape}")
                
                # 假设每个元素的形状为 [sequence_length, num_experts]
                # 这里需要确认 num_experts 的数量
                num_experts = example_shape[1]  # 从形状推断 num_experts
                
                # 对每层的 router_logits 进行 argmax 操作，获取选择的专家索引
                # 这里每个 layer 的 router_logits 是 [sequence_length, num_experts]
                # 进行 argmax 后，每个 layer 的选择是 [sequence_length]
                selected_experts = [logits.argmax(dim=-1) for logits in router_logits_tuple]  # list of [sequence_length]
                
                # stack 成 [sequence_length, num_layers]
                selected_experts = torch.stack(selected_experts, dim=-1)  # [sequence_length, num_layers]
                print("selected_experts shape:", selected_experts.shape)
                
                print(f"\n请求 {req_idx + 1}: {req}")
                print("每一层中激活的专家数量以及请求中激活的专家总数:")
                
                sequence_length, num_layers = selected_experts.shape
                
                # 初始化一个集合来存储整个请求中激活的专家
                total_activated_experts = set()
                num_total = 0

                for layer_idx in range(num_layers):
                    # 获取该层所有 token 选择的专家
                    experts_in_layer = selected_experts[:, layer_idx].tolist()
                    # 使用 set 去重
                    unique_experts_in_layer = set(experts_in_layer)
                    num_unique_experts = len(unique_experts_in_layer)
                    num_total += num_unique_experts
                    print(f"  Layer {layer_idx}: 激活的专家数量 = {num_unique_experts}")
                    
                    # 将该层的专家加入到总集合中
                    total_activated_experts.update(unique_experts_in_layer)
                
                print(f"  该请求中激活的专家总数 = {num_total}\n")
                
                # 将 num_total 写入 activated_experts.txt 文件，确保每行只有一个数字
                output_file.write(f"{num_total}\n")
    
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
