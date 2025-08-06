import torch.nn as nn
from attention import MultiHeadAttention

class Encoder(nn.Module):

    def __init__(self, d_model, n_heads, dropout = 0.1, ratio = 4):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attention = MultiHeadAttention(d_model, n_heads)    # self-attention layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*ratio, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model) # norm after attention
        self.norm2 = nn.LayerNorm(d_model) # norm after feedforward
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        norm1_x = self.norm1(x)
        x = x + self.dropout(self.self_attention(norm1_x, norm1_x, norm1_x))
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x