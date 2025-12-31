"""
Training module for RAT-DoA model
"""

import torch
import torch.optim as optim
from torch.nn import L1Loss
import numpy as np
from pathlib import Path

class RAT_Trainer:
    """Train RAT model on pre-extracted features"""
    
    def __init__(self, model, device='cpu', save_dir='results/models'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = L1Loss()  # MAE loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10, verbose=True
        )
        
        self.history = {'train_loss': [], 'val_mae': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_fn(predictions.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        predictions_list = []
        labels_list = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features)
                predictions_list.extend(predictions.cpu().squeeze().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        mae = np.mean(np.abs(np.array(predictions_list) - np.array(labels_list)))
        return mae, predictions_list, labels_list
    
    def train(self, train_loader, val_loader, epochs=100, patience=20):
        """Full training loop"""
        best_val_mae = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_mae, _, _ = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_mae'].append(val_mae)
            
            self.scheduler.step(val_mae)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val MAE={val_mae:.4f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                self.save_checkpoint(f'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        filepath = self.save_dir / filename
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = self.save_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint: {filepath}")
