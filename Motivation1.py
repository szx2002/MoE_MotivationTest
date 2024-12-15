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

        total_transformer_runtime = 0.0
        normal_transformer_runtime = 0.0
        first_normal_layer_name = None
        
        layer_start_events = {}

        def make_pre_hook(module_id):
            def forward_pre_hook(module, inp):
                # 创建并记录开始事件
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                layer_start_events[module_id] = start_event
            return forward_pre_hook

        def make_post_hook(module_id, layer_name):
            def forward_hook(module, inp, out):
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                # 等待end_event完成
                end_event.synchronize()
                start_event = layer_start_events[module_id]
                elapsed_ms = start_event.elapsed_time(end_event)  # 毫秒
                elapsed = elapsed_ms / 1000.0  # 转换成秒

                nonlocal total_transformer_runtime, normal_transformer_runtime
                nonlocal first_normal_layer_name

                total_transformer_runtime += elapsed
                normal_transformer_runtime += elapsed
                if first_normal_layer_name is None:
                    first_normal_layer_name = layer_name
            return forward_hook

        # 为每一层中的子模块注册hook（只为普通层）
        def register_hooks_for_submodules(parent_module, parent_name=""):
            for name, sub_module in parent_module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                sub_module.register_forward_pre_hook(make_pre_hook(id(sub_module)))
                sub_module.register_forward_hook(make_post_hook(id(sub_module), full_name))
                register_hooks_for_submodules(sub_module, full_name)

        for i, layer_module in enumerate(model.model.layers):
            layer_name = f"layer_{i}"
            register_hooks_for_submodules(layer_module, layer_name)

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

            total_transformer_runtime = 0.0
            normal_transformer_runtime = 0.0
            first_normal_layer_name = None
            layer_start_events.clear()
            
            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            # 使用CUDA事件计时整体预测时间
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)

            total_start.record()
            with torch.inference_mode():
                # 假设 outputs 中包含 'latency' 键
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    use_cache=False,
                )
                print(outputs)
                 
            total_end.record()
            total_end.synchronize()

            total_runtime_ms = total_start.elapsed_time(total_end)
            total_runtime = total_runtime_ms / 1000.0

            # 使用 outputs['latency'] 作为 MoE 层的 latency
            moe_transformer_runtime = outputs.get("moe_latency", 0.0)

            moe_times.append(moe_transformer_runtime)
            normal_times.append(normal_transformer_runtime)
            total_times.append(total_runtime)

            print(f"请求长度 {req_length} 完成：")
            print(f"  第一个普通层: {first_normal_layer_name if first_normal_layer_name else '无普通层'}")
            print(f"  使用outputs中latency记录的MoE层运行时间: {moe_transformer_runtime:.4f}s")
            print(f"  普通层运行时间: {normal_transformer_runtime:.4f}s")
            print(f"  整个预测过程运行时间: {total_runtime:.4f}s")

        plt.figure(figsize=(10, 6))
        plt.plot(request_lengths, moe_times, label='MoE Layers Runtime (from outputs["moe_latency"])', marker='o')
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
