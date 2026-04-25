from google import genai
import torch

class MedicalEmbeddingClient:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def get_embedding(self, text):
        response = self.client.models.embed_content(
            model="gemini-embedding-2", 
            contents=str(text)
        )
        return torch.tensor([response.embeddings[0].values])


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