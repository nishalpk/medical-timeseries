import torch
from models.moe_system import FullSystemMoE
from utils.gemini_client import MedicalEmbeddingClient

# Setup
API_KEY = "YOUR_NEW_KEY"
gemini = MedicalEmbeddingClient(API_KEY)
model = FullSystemMoE(text_embed_dim=3072).eval()

# Mock Inputs
graphrag_data = {"condition": "sepsis", "vitals": "norepinephrine"}
vitals_data = torch.randn(1, 48, 5)

# Run
with torch.no_grad():
    text_emb = gemini.get_embedding(graphrag_data)
    risk, forecast, weights = model(vitals_data, text_emb)

print(f"Risk: {risk.item()*100:.2f}% | Experts: {weights.tolist()}")
