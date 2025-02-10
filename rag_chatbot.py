import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
from api_handler import get_api_key  # Import API key handler

# Configurations
vector_db_file = "vector_store.faiss"
metadata_file = "metadata.txt"
model_name = "all-MiniLM-L6-v2"
top_k = 5
llama_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# Load API key
api_key = get_api_key()

# Initialize embedding model
embedding_model = SentenceTransformer(model_name)

def load_metadata(metadata_file):
    with open(metadata_file, 'r', encoding='utf-8') as meta_file:
        metadata = meta_file.readlines()
    return metadata

def retrieve_chunks(query, db_file, metadata_file, top_k=5):
    index = faiss.read_index(db_file)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    metadata = load_metadata(metadata_file)

    chunks = []
    for idx in indices[0]:
        if idx != -1:
            _, chunk = metadata[idx].strip().split("\t", 1)
            chunks.append(chunk)
    return chunks

def generate_response(query, chunks):
    """Generate an answer using LLaMA 70B API."""
    context = " ".join(chunks)
    payload = {
        "model": "meta/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}],
        "temperature": 0.5,
        "max_tokens": 8192
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.post(llama_api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    print("Enter your question:")
    user_query = input("> ")

    print("Retrieving relevant chunks...")
    relevant_chunks = retrieve_chunks(user_query, vector_db_file, metadata_file, top_k)

    print("Generating response...")
    answer = generate_response(user_query, relevant_chunks)

    print("\nAnswer:")
    print(answer)
