from google import genai
import torch

class MedicalEmbeddingClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_embedding(self, text):
        return torch.randn(1, 3072)


# if __name__ == "__main__":
#     import os
#     from dotenv import load_dotenv

#     load_dotenv()
#     gemini_key = os.getenv("GEMINI_API")
    
#     client = MedicalEmbeddingClient(gemini_key)
#     test_text = "Patient suspected Sepsis -> Blood Culture Ordered -> Antibiotics Administered"
#     embedding = client.get_embedding(test_text)
    
#     print(f"Text: {test_text}")
#     print(f"Embedding Shape: {embedding.shape}")
#     print(f"Embedding Sample Values: {embedding[0][:5]}")