
trainer.py
"""
RAT-DoA Model Trainer
Complete training loop with validation, early stopping, and checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import json


class RAT_Trainer:
    """
    Trainer class for RAT-DoA model
    Handles: training loop, validation, early stopping, model checkpointing
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        Initialize trainer
        
        Args:
            model: RAT_DoA model
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler for learning rate decay
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)  # [B] -> [B, 1]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, List, List]:
        """
        Validate on validation set
        
        Returns:
            mae: Mean absolute error (normalized scale)
            predictions: Model predictions
            labels: Ground truth labels
        """
        self.model.eval()
        total_mae = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(batch_x)
                mae = torch.mean(torch.abs(predictions - batch_y))
                
                total_mae += mae.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy().flatten().tolist())
                all_labels.extend(batch_y.cpu().numpy().flatten().tolist())
        
        avg_mae = total_mae / num_batches
        return avg_mae, all_predictions, all_labels
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = 100,
              patience: int = 15,
              save_dir: str = 'results/models') -> Dict:
        """
        Full training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_dir: Directory to save best model
            
        Returns:
            history: Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = save_dir / 'best_model.pt'
        patience_counter = 0
        
        print(f"\nStarting training on {self.device}")
        print(f"Total epochs: {epochs}, Early stopping patience: {patience}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_mae, _, _ = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val MAE: {val_mae:.6f} | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping & model saving
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved! MAE: {val_mae:.6f}\n")
                
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best model from epoch {self.best_epoch} with MAE: {self.best_val_mae:.6f}")
                    break
            
            # Learning rate scheduling
            self.scheduler.step(val_mae)
        
        # Load best model
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"\n✓ Loaded best model from epoch {self.best_epoch}")
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on dataset
        
        Args:
            data_loader: Data loader
            
        Returns:
            predictions: Model predictions
            labels: Ground truth labels
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                
                all_predictions.extend(predictions.cpu().numpy().flatten().tolist())
                all_labels.extend(batch_y.cpu().numpy().flatten().tolist())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'learning_rate': self.learning_rate,
            'best_val_mae': float(self.best_val_mae),
            'best_epoch': self.best_epoch
        }