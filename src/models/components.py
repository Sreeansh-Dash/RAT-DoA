"""
RAT Model Components - ADAPTED FOR 40 FEATURES
No longer uses 1D Conv (features are already extracted)
"""

import torch
import torch.nn as nn
import math

class ResNetBlock(nn.Module):
    """Residual block for dense features (not 1D Conv)"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut for dimension matching
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.relu(x)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        return output

class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.self_attn(x))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence position awareness"""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
