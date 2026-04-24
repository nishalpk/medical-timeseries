import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import MIRATimeSeriesEncoder
from .experts import ResidualClinicalExpert

class LLMGatingNetwork(nn.Module):
    def __init__(self, text_embedding_dim, num_experts=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, llm_embeddings):
        return F.softmax(self.gate(llm_embeddings), dim=-1)

class FullSystemMoE(nn.Module):
    def __init__(self, time_features=5, time_latent_dim=128, text_embed_dim=3072):
        super().__init__()
        self.time_series_encoder = MIRATimeSeriesEncoder(time_features, time_latent_dim)
        self.llm_router = LLMGatingNetwork(text_embed_dim, num_experts=3)
        self.hemodynamic_expert = ResidualClinicalExpert(time_latent_dim)
        self.biochemical_expert = ResidualClinicalExpert(time_latent_dim)
        self.generalist_expert = ResidualClinicalExpert(time_latent_dim)
        self.risk_classifier = nn.Sequential(nn.Linear(time_latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))    
        self.forecaster = nn.Linear(time_latent_dim, 24 * time_features) 
        self.time_features = time_features

    def forward(self, raw_time_series, graphrag_llm_embeddings):
        neural_signals = self.time_series_encoder(raw_time_series)
        weights = self.llm_router(graphrag_llm_embeddings) 
        experts_out = torch.stack([self.hemodynamic_expert(neural_signals), 
                                   self.biochemical_expert(neural_signals), 
                                   self.generalist_expert(neural_signals)], dim=1)
        fused = torch.bmm(weights.unsqueeze(1), experts_out).squeeze(1)
        return torch.sigmoid(self.risk_classifier(fused)), \
               self.forecaster(fused).view(-1, 24, self.time_features), weights
