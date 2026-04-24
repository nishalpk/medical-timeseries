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
