import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from huggingface_hub import login

# 登录 Token
token = "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"
print("Logging in to Hugging Face Hub...")
login(token)
print("Login complete.")

# 分词器和模型路径
tokenizer_path = "/vllm-workspace/huggingfaceM87Bv01/Mixtral-8x7B-v0.1"
requests_file = "requests.txt"

print(f"Checking if {requests_file} exists...")
if not os.path.exists(requests_file):
    raise FileNotFoundError(f"Error: File '{requests_file}' not found.")
print("requests.txt found.")

print(f"Checking if tokenizer path {tokenizer_path} exists...")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Error: Tokenizer folder '{tokenizer_path}' not found.")
print("Tokenizer folder found.")

def read_requests(file_path):
    print(f"Reading requests from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Total {len(lines)} requests read.")
    return lines

def get_token_embeddings(request, tokenizer, model):
    inputs = tokenizer(request, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
    return token_embeddings

# 将请求矩阵统一padding到相同大小，然后展平为向量
def pad_and_flatten(embeddings_list):
    max_len = max(e.shape[0] for e in embeddings_list)
    embed_dim = embeddings_list[0].shape[1]
    print(f"Maximum token length among all requests: {max_len}")
    print(f"Embedding dimension: {embed_dim}")

    padded_embeddings = []
    for i, emb in enumerate(embeddings_list):
        length = emb.shape[0]
        if length < max_len:
            pad_tensor = torch.zeros((max_len - length, embed_dim), dtype=emb.dtype, device=emb.device)
            emb = torch.cat([emb, pad_tensor], dim=0)
        flattened = emb.view(-1)
        padded_embeddings.append(flattened)
    return padded_embeddings

def vector_distance_and_similarity(vec1, vec2):
    # 欧氏距离
    euclidean_distance = torch.dist(vec1, vec2).item()
    # 余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=-1).item()
    return euclidean_distance, cosine_similarity

if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token)
    print("Tokenizer loaded.")

    print("Loading model (this may take a while if the model is large or not cached)...")
    model = AutoModel.from_pretrained(tokenizer_path, token=token)
    print("Model loaded.")

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to GPU.")
    else:
        print("CUDA not available. Using CPU.")
    
    requests = read_requests(requests_file)

    # 提取所有请求的token矩阵
    requests_embeddings = []
    print("Extracting embeddings for each request...")
    for i, req in enumerate(requests):
        print(f"Processing request {i+1}/{len(requests)}: {req}")
        emb = get_token_embeddings(req, tokenizer, model)
        print(f"Request {i+1}: got embeddings with shape {emb.shape}")
        requests_embeddings.append(emb)
    print("All requests processed.")

    # 将所有request的embedding矩阵pad成相同长度并flatten
    print("Padding and flattening embeddings...")
    flattened_embeddings = pad_and_flatten(requests_embeddings)
    num_requests = len(flattened_embeddings)

    print("Calculating distance matrices...")
    euclidean_dist_matrix = np.zeros((num_requests, num_requests))
    cosine_sim_matrix = np.zeros((num_requests, num_requests))

    for i in range(num_requests):
        for j in range(num_requests):
            if i == j:
                euclidean_dist_matrix[i, j] = 0.0
                cosine_sim_matrix[i, j] = 1.0
            elif j > i:
                dist, sim = vector_distance_and_similarity(flattened_embeddings[i], flattened_embeddings[j])
                euclidean_dist_matrix[i, j] = dist
                euclidean_dist_matrix[j, i] = dist
                cosine_sim_matrix[i, j] = sim
                cosine_sim_matrix[j, i] = sim

    print("Request-level Matrix-based Distance:")
    print("Euclidean Distance Matrix:")
    print(euclidean_dist_matrix)
    print("\nCosine Similarity Matrix:")
    print(cosine_sim_matrix)

    print("Computation finished.")
