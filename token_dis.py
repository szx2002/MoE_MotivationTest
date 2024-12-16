import os
from transformers import AutoTokenizer
from huggingface_hub import login
import numpy as np

# 分词器路径
tokenizer_path = "huggingfaceM87Bv01"
requests_file = "requests.txt"
token = "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"
login(token)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
            "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1",
            token=token
        )

# 读取 requests.txt 文件
def read_requests(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# 计算 token 之间距离的平均值
def calculate_average_distance(tokens):
    if len(tokens) < 2:
        return 0  # 如果 token 少于 2 个，距离为 0
    distances = [tokens[i + 1] - tokens[i] for i in range(len(tokens) - 1)]
    return np.mean(distances)

# 处理每个请求并输出结果
def process_requests(requests, tokenizer):
    for i, request in enumerate(requests):
        # 对请求进行分词
        tokens = tokenizer.encode(request, add_special_tokens=False)
        token_count = len(tokens)
        avg_distance = calculate_average_distance(tokens)
        
        print(f"Request {i + 1}:")
        print(f"  Token Count: {token_count}")
        print(f"  Average Token Distance: {avg_distance:.2f}\n")

if __name__ == "__main__":
    # 检查文件和分词器是否存在
    if not os.path.exists(requests_file):
        print(f"Error: File '{requests_file}' not found.")
    elif not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer folder '{tokenizer_path}' not found.")
    else:
        requests = read_requests(requests_file)
        process_requests(requests, tokenizer)
