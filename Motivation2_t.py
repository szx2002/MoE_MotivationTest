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

        # 存储每一层的 router_logits
        layer_router_logits = defaultdict(list)

        def hook_fn(layer_name):
            def hook(module, inputs, outputs):
                if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                    layer_router_logits[layer_name].append(outputs.router_logits)
            return hook

        # 为模型的每一层注册 Hook
        for i, layer in enumerate(model.model.layers):
            layer.register_forward_hook(hook_fn(f"layer_{i}"))

        input_file = "requests.txt"
        requests = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                req = line.strip()
                if req:
                    requests.append(req)

        # 仅处理前两个请求
        for req_idx, req in enumerate(requests[:2]):
            layer_router_logits.clear()
            inputs = tokenizer(req, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict=True,
                    output_router_logits=True  # 确保输出 router_logits
                )

            print(f"\n请求 {req_idx + 1}: {req}")
            print("每一层的 token 选择的 Experts:")

            for layer_name, logits_list in layer_router_logits.items():
                for token_idx, router_logits in enumerate(logits_list[0].argmax(dim=-1)):
                    token = tokenizer.decode(inputs.input_ids[0][token_idx])
                    expert = router_logits.item()
                    print(f"{layer_name}, Token: '{token}' -> Expert: {expert}")

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