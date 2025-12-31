"""
RAT Model for 479-feature EEG data
With feature embedding and dimension reduction
"""

import torch
import torch.nn as nn
from .components import ResNetBlock, TransformerEncoderLayer, PositionalEncoding

class RAT_DoA(nn.Module):
    """
    ResNet-Attention-Transformer for 479 EEG features
    
    Strategy: Compress 479 → 256 → process with Transformer
    """
    
    def __init__(self,
                 input_dim: int = 479,
                 embedding_dim: int = 256,      # Compress to 256
                 resnet_dims: list = [512, 1024],
                 attention_heads: int = 8,
                 attention_dim: int = 256,
                 transformer_layers: int = 3,
                 transformer_dim_ff: int = 512,
                 dropout: float = 0.2,           # Higher dropout for large feature set
                 output_dim: int = 1):
        
        super().__init__()
        
        # ===== Feature Embedding (479 → 256) =====
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # ===== ResNet Backbone =====
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(embedding_dim, resnet_dims, dropout),
            ResNetBlock(resnet_dims, resnet_dims, dropout),
        )
        
        final_dim = resnet_dims[-1]  # 1024
        
        # ===== Create sequence for Transformer =====
        # Map 1024 features → 8 tokens of 128-dim each
        self.seq_length = 8
        self.to_seq = nn.Linear(final_dim, attention_dim * self.seq_length)
        
        # ===== Positional Encoding =====
        self.pos_encoding = PositionalEncoding(attention_dim)
        
        # ===== Transformer Encoder =====
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                attention_dim,
                num_heads=attention_heads,
                dim_feedforward=transformer_dim_ff,
                dropout=dropout
            )
            for _ in range(transformer_layers)
        ])
        
        # ===== Regression Head =====
        self.regression_head = nn.Sequential(
            nn.Linear(attention_dim * self.seq_length, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, 479] features
        
        Returns:
            predictions: [batch, 1] BIS values
        """
        batch_size = x.size(0)
        
        # ===== Feature Embedding =====
        x = self.feature_embedding(x)      # [B, 256]
        
        # ===== ResNet =====
        x = self.resnet_blocks(x)          # [B, 1024]
        
        # ===== Reshape for Transformer =====
        x = self.to_seq(x)                 # [B, 256*8]
        x = x.reshape(batch_size, self.seq_length, -1)  # [B, 8, 256]
        
        # ===== Add Positional Encoding =====
        x = self.pos_encoding(x)           # [B, 8, 256]
        
        # ===== Transformer Encoder =====
        for layer in self.transformer_layers:
            x = layer(x)                   # [B, 8, 256]
        
        # ===== Flatten for Regression =====
        x = x.reshape(batch_size, -1)      # [B, 2048]
        
        # ===== Regression Head =====
        predictions = self.regression_head(x)  # [B, 1]
        
        return predictions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = RAT_DoA(input_dim=479)
    print(f"RAT-DoA (479-feature version)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    dummy = torch.randn(32, 479)
    out = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {out.shape}")
    print("✓ Model test successful!")
