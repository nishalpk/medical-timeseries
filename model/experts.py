import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualClinicalExpert(nn.Module):
    """Residual block for clinical specialization."""
    def __init__(self, latent_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim) 
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.gelu(out + identity)
