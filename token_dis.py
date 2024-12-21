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
requests_file = "requestM3.txt"
output_file = "token_dis.txt"  # 输出文件名

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=token)
model = AutoModel.from_pretrained(tokenizer_path, use_auth_token=token)

# 读取 requests.txt 文件
def read_requests(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# 计算相邻 token 向量之间的欧几里得距离和余弦相似度
def calculate_distances_and_similarity(embeddings):
    if len(embeddings) < 2:
        return 0, 0  # 如果 token 少于 2 个，返回 0
    
    euclidean_distances = [torch.dist(embeddings[i], embeddings[i + 1]).item() for i in range(len(embeddings) - 1)]
    cosine_similarities = [
        torch.nn.functional.cosine_similarity(embeddings[i], embeddings[i + 1], dim=0).item() 
        for i in range(len(embeddings) - 1)
    ]
    
    avg_euclidean_distance = np.mean(euclidean_distances)
    avg_cosine_similarity = np.mean(cosine_similarities)
    
    return avg_euclidean_distance, avg_cosine_similarity

# 处理每个请求并输出结果到 token_dis.txt
def process_requests(requests, tokenizer, model, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, request in enumerate(requests):
            # 对请求进行分词并获取 token 向量
            inputs = tokenizer(request, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)
            
            token_count = token_embeddings.shape[0]
            avg_euclidean_distance, avg_cosine_similarity = calculate_distances_and_similarity(token_embeddings)
            
            # 写入文件，每行包含两个数值，以空格分隔
            f_out.write(f"{avg_euclidean_distance} {avg_cosine_similarity}\n")
            
            # 可选：打印进度
            print(f"Processed Request {i + 1}: Avg Euclidean Distance = {avg_euclidean_distance:.4f}, Avg Cosine Similarity = {avg_cosine_similarity:.4f}")

if __name__ == "__main__":
    # 检查文件和分词器是否存在
    if not os.path.exists(requests_file):
        print(f"Error: File '{requests_file}' not found.")
    elif not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer folder '{tokenizer_path}' not found.")
    else:
        requests = read_requests(requests_file)
        process_requests(requests, tokenizer, model, output_file)
        print(f"Results have been written to '{output_file}'.")
