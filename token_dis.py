import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from huggingface_hub import login

# 登录 Token
token = "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"
login(token)

# 分词器和模型路径
tokenizer_path = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
requests_file = "requests.txt"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token)
model = AutoModel.from_pretrained(tokenizer_path, token=token)

# 读取 requests.txt 文件
def read_requests(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# 计算相邻 token 向量之间的欧几里得距离的平均值
def calculate_average_vector_distance(embeddings):
    if len(embeddings) < 2:
        return 0  # 如果 token 少于 2 个，距离为 0
    distances = [torch.dist(embeddings[i], embeddings[i + 1]).item() for i in range(len(embeddings) - 1)]
    return np.mean(distances)

# 处理每个请求并输出结果
def process_requests(requests, tokenizer, model):
    for i, request in enumerate(requests):
        # 对请求进行分词并获取 token 向量
        inputs = tokenizer(request, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        
        token_count = token_embeddings.shape[0]
        avg_distance = calculate_average_vector_distance(token_embeddings)
        
        print(f"Request {i + 1}:")
        print(f"  Token Count: {token_count}")
        print(f"  Average Token Distance: {avg_distance:.4f}\n")

if __name__ == "__main__":
    # 检查文件和分词器是否存在
    if not os.path.exists(requests_file):
        print(f"Error: File '{requests_file}' not found.")
    elif not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer folder '{tokenizer_path}' not found.")
    else:
        requests = read_requests(requests_file)
        process_requests(requests, tokenizer, model)
