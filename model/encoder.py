import torch
import torch.nn as nn


## Trash: Placeholder for the MIRA Time-Series Encoder (Bi-GRU + Attention)
class MIRATimeSeriesEncoder(nn.Module):
    """Processes raw ICU vitals using Bi-GRU and Attention."""
    def __init__(self, input_features=5, hidden_dim=128):
        super().__init__()
        self.bigru = nn.GRU(input_features, hidden_dim // 2, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        rnn_out, _ = self.bigru(x)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        pooled, _ = torch.max(attn_out, dim=1) 
        return self.layer_norm(pooled)
