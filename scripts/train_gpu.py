#!/usr/bin/env python3

"""
PRODUCTION ENSEMBLE TRAINING SCRIPT
====================================
Trains 3 independent RAT-DoA models with realistic expectations
- Minimal augmentation (only Gaussian noise for regularization)
- Smart hyperparameter tuning per model
- Conservative learning rates with early stopping
- Per-model checkpointing + validation tracking
- GPU optimized, realistic 4-5 hour training
- NO data tricks - pure learning capacity

Expected Results (0-100 BIS scale):
- MAE: 3.0-3.5 (realistic, publication-viable)
- R²: 0.75-0.85 (solid performance)
- Pearson r: 0.70-0.80 (good correlation)

Compatible with evaluate.py - saves to results/models/ensemble_model_*.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path
import json
import time
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.data.preprocessor import LargeFeaturePreprocessor
from src.models.rat_model import RAT_DoA, count_parameters


# ============================================================================
# MINIMAL AUGMENTATION (Regularization only)
# ============================================================================

class MinimalAugmentor:
    """Light augmentation - only Gaussian noise for regularization"""
    
    def __init__(self, noise_sigma=0.03):
        """
        Light noise injection to prevent overfitting
        sigma=0.03 means ~3% of feature variance
        """
        self.noise_sigma = noise_sigma
    
    @staticmethod
    def add_gaussian_noise(X, sigma=0.03):
        """Add Gaussian noise - real-world measurement uncertainty"""
        if random.random() < 0.5:  # 50% probability
            noise = torch.randn_like(X) * sigma
            return X + noise
        return X
    
    def augment_batch(self, X, y):
        """Minimal augmentation - only noise"""
        X = self.add_gaussian_noise(X, self.noise_sigma)
        return X, y


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

class ConservativeScheduler:
    """Conservative learning rate: warmup + exponential decay"""
    
    def __init__(self, optimizer, warmup_epochs=8, total_epochs=150, base_lr=5e-4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup: 0 -> base_lr
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Exponential decay after warmup
            decay_epochs = self.current_epoch - self.warmup_epochs
            total_decay = self.total_epochs - self.warmup_epochs
            lr = self.base_lr * (0.1 ** (decay_epochs / total_decay))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, 1e-6)  # Floor at 1e-6
        
        return lr


# ============================================================================
# PRODUCTION TRAINER (Single Model)
# ============================================================================

class ProductionTrainer:
    """Production-grade trainer with realistic expectations"""
    
    def __init__(self, model, device='cuda', model_id=1, model_type='standard'):
        """
        Args:
            model: RAT_DoA model instance
            device: 'cuda' or 'cpu'
            model_id: 1, 2, or 3 (for ensemble)
            model_type: 'standard', 'large', or 'xlarge'
        """
        self.model = model.to(device)
        self.device = device
        self.model_id = model_id
        self.model_type = model_type
        
        # Optimizer: AdamW with conservative weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-4,  # Conservative base learning rate
            weight_decay=1e-5,  # Light L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Loss: Smooth L1 (Huber) for robustness to outliers
        self.criterion = nn.SmoothL1Loss(beta=0.1)
        
        # Minimal augmentation
        self.augmentor = MinimalAugmentor(noise_sigma=0.03)
        
        # History tracking
        self.train_losses = deque(maxlen=100)
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        self.start_time = None
    
    def train_epoch(self, train_loader, scheduler):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Minimal augmentation (noise only)
            X_batch, y_batch = self.augmentor.augment_batch(X_batch, y_batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch).squeeze()
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            self.train_losses.append(loss.item())
            
            # Progress per 20% of epoch
            if (batch_idx + 1) % max(5, len(train_loader) // 5) == 0:
                avg_loss = np.mean(list(self.train_losses))
                print(f"      Model {self.model_id} | Batch {batch_idx+1:3d}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.6f} | LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step()
        return epoch_loss / num_batches
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs=150, patience=40):
        """Full training loop with early stopping"""
        self.start_time = time.time()
        
        scheduler = ConservativeScheduler(
            self.optimizer,
            warmup_epochs=8,
            total_epochs=epochs,
            base_lr=5e-4
        )
        
        print(f"\n{'='*90}")
        print(f"🚀 TRAINING MODEL {self.model_id} ({self.model_type.upper()})")
        print(f"{'='*90}")
        print(f"Epochs: {epochs} | Patience: {patience} | Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"Optimizer: AdamW (lr=5e-4, wd=1e-5)")
        print(f"Loss: Smooth L1 (Huber, beta=0.1) - robust to outliers")
        print(f"Augmentation: Gaussian noise only (sigma=0.03)")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, scheduler)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            elapsed = time.time() - self.start_time
            
            # Track best model
            status = "   "
            if val_loss < self.best_val_loss:
                status = "🌟"
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter > 0 and self.patience_counter % 15 == 0:
                    status = f"⚠️ "
            
            # Detailed progress every epoch
            print(f"Epoch {epoch:3d}/{epochs} | {status} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {self.best_val_loss:.6f}@E{self.best_epoch} | "
                  f"Patience: {self.patience_counter:2d}/{patience} | "
                  f"Elapsed: {elapsed/60:5.1f}m")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch} (patience {patience} reached)")
                break
        
        # Load best checkpoint
        self._load_best_checkpoint()
        elapsed = time.time() - self.start_time
        
        print(f"\n✨ Model {self.model_id} complete!")
        print(f"   Best epoch: {self.best_epoch}")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        print(f"   Total time: {elapsed/60:.1f}m ({elapsed/3600:.2f}h)")
        
        return {
            'train_losses': list(self.train_losses),
            'val_losses': self.val_losses,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'total_time': elapsed
        }
    
    def _save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        ckpt_dir = Path('results/models')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f'ensemble_model_{self.model_id}.pt'
        torch.save(self.model.state_dict(), ckpt_path)
    
    def _load_best_checkpoint(self):
        """Load best checkpoint"""
        ckpt_path = Path('results/models') / f'ensemble_model_{self.model_id}.pt'
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"   ✓ Loaded: {ckpt_path}")


# ============================================================================
# MODEL ARCHITECTURES (Different capacities)
# ============================================================================

def create_standard_model(input_dim):
    """Standard capacity - balanced performance/speed"""
    return RAT_DoA(
        input_dim=input_dim,
        embedding_dim=192,
        resnet_dims=[384, 384],
        attention_heads=6,
        transformer_layers=2,
        dropout=0.15,
    )


def create_large_model(input_dim):
    """Large capacity - better feature learning"""
    return RAT_DoA(
        input_dim=input_dim,
        embedding_dim=256,
        resnet_dims=[512, 512],
        attention_heads=8,
        transformer_layers=2,
        dropout=0.1,
    )


def create_xlarge_model(input_dim):
    """XLarge capacity - maximum capacity"""
    return RAT_DoA(
        input_dim=input_dim,
        embedding_dim=320,
        resnet_dims=[640, 640],
        attention_heads=10,
        transformer_layers=3,
        dropout=0.08,
    )


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("\n" + "="*90)
    print("🎯 PRODUCTION ENSEMBLE TRAINING (3 Models)")
    print("="*90)
    print("Realistic expectations: MAE 3.0-3.5, R² 0.75-0.85, Pearson r 0.70-0.80")
    print("Publication-viable results without aggressive augmentation")
    
    # ===== LOAD DATA =====
    print("\n1️⃣  Loading data (33,435 samples × 479 features)...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    
    # ===== SPLIT DATA =====
    print("2️⃣  Splitting data (70% train, 15% val, 15% test)...")
    train_idx, val_idx, test_idx = loader.prepare_data()
    
    # ===== LOAD FEATURES =====
    print("3️⃣  Loading features...")
    X_train, y_train = loader.get_feature_subset(train_idx)
    X_val, y_val = loader.get_feature_subset(val_idx)
    X_test, y_test = loader.get_feature_subset(test_idx)
    
    print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    # ===== PREPROCESS =====
    print("4️⃣  Preprocessing (normalization + outlier removal)...")
    prep = LargeFeaturePreprocessor()
    
    X_train, train_params = prep.normalize_features(X_train)
    X_val, _ = prep.normalize_features(X_val, train_params['means'], train_params['stds'])
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    
    y_train, bis_params = prep.normalize_bis(y_train)
    y_val, _ = prep.normalize_bis(y_val, bis_params['mean'], bis_params['std'])
    y_test, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    # Remove outliers from training set only
    X_train_clean, y_train_clean = prep.remove_outliers(X_train, y_train)
    print(f"   Removed {len(X_train) - len(X_train_clean)} outliers")
    print(f"   Clean train set: {X_train_clean.shape}")
    
    X_train = X_train_clean
    y_train = y_train_clean
    
    # ===== CREATE DATALOADERS =====
    print("5️⃣  Creating DataLoaders...")
    
    batch_size = 32  # GPU-friendly
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # ===== DEVICE SETUP =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n6️⃣  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
    
    # ===== TRAIN ENSEMBLE (3 Models) =====
    input_dim = X_train.shape[1]
    
    print(f"\n7️⃣  Training 3 Models (Expected: 4-5 hours)")
    print(f"   Input dim: {input_dim}")
    
    # Model configurations for diversity
    model_configs = [
        ('standard', create_standard_model),
        ('large', create_large_model),
        ('xlarge', create_xlarge_model),
    ]
    
    training_histories = []
    
    for model_id, (model_type, create_fn) in enumerate(model_configs, 1):
        # Create fresh model
        model = create_fn(input_dim)
        
        # Train
        trainer = ProductionTrainer(
            model,
            device=device,
            model_id=model_id,
            model_type=model_type
        )
        
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=150,
            patience=40
        )
        
        training_histories.append(history)
        
        # Save training history
        history_path = Path('results/training') / f'model_{model_id}_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_to_save = {
            'model_type': model_type,
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val_loss'],
            'total_epochs_trained': len(history['val_losses']),
            'total_time_hours': history['total_time'] / 3600,
            'val_losses_final_10': history['val_losses'][-10:]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        
        print(f"   ✓ History saved: {history_path}")
    
    # ===== TEST ENSEMBLE =====
    print(f"\n8️⃣  Testing Ensemble on Test Set...")
    
    all_test_preds = []
    
    for model_id in range(1, 4):
        model_path = Path('results/models') / f'ensemble_model_{model_id}.pt'
        
        if not model_path.exists():
            print(f"   ⚠️  Model {model_id} not found: {model_path}")
            continue
        
        model_config = model_configs[model_id - 1]
        model = model_config[1](input_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch).squeeze().cpu().numpy()
                preds.extend(pred)
        
        all_test_preds.append(np.array(preds))
        print(f"   ✓ Model {model_id} ({model_config[0]}): {len(preds)} predictions")
    
    # Ensemble: Average of 3 models
    if len(all_test_preds) == 3:
        ensemble_pred = np.mean(all_test_preds, axis=0)
        print(f"   ✓ Ensemble: Average of {len(all_test_preds)} models")
    else:
        print(f"   ⚠️  Only {len(all_test_preds)} models available")
        if len(all_test_preds) == 0:
            return
        ensemble_pred = np.mean(all_test_preds, axis=0)
    
    # Denormalize
    ensemble_pred_orig = prep.denormalize_bis(ensemble_pred, bis_params)
    test_labels_orig = prep.denormalize_bis(y_test, bis_params)
    
    # ===== COMPUTE METRICS =====
    mae = np.mean(np.abs(ensemble_pred_orig - test_labels_orig))
    rmse = np.sqrt(np.mean((ensemble_pred_orig - test_labels_orig) ** 2))
    
    ss_res = np.sum((test_labels_orig - ensemble_pred_orig) ** 2)
    ss_tot = np.sum((test_labels_orig - test_labels_orig.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    pearson_r, pearson_p = pearsonr(ensemble_pred_orig, test_labels_orig)
    
    mape = np.mean(np.abs((test_labels_orig - ensemble_pred_orig) / (test_labels_orig + 1e-6))) * 100
    
    # ===== FINAL REPORT =====
    print(f"\n" + "="*90)
    print(f"✨ TRAINING COMPLETE!")
    print(f"="*90)
    
    print(f"\n📊 TEST SET RESULTS (0-100 BIS Scale):")
    print(f"   MAE:         {mae:.4f}")
    print(f"   RMSE:        {rmse:.4f}")
    print(f"   R²:          {r2:.4f}")
    print(f"   Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"   MAPE:        {mape:.2f}%")
    print(f"   Test set:    {len(test_labels_orig)} samples")
    
    print(f"\n🎯 PUBLICATION VIABILITY:")
    if mae < 3.0:
        print(f"   ✅ EXCELLENT (MAE < 3.0) - Strong publication candidate!")
        print(f"      Target: IEEE Sensors Journal, Biomedical Engineering journals")
    elif mae < 3.5:
        print(f"   ✅ GREAT (MAE < 3.5) - Publication viable!")
        print(f"      Target: Medical journals, Sensors (MDPI)")
    elif mae < 4.0:
        print(f"   👍 GOOD (MAE < 4.0) - Competitive with similar work")
        print(f"      Target: Conference papers, domain journals")
    else:
        print(f"   ⚠️  Consider further optimization or feature engineering")
    
    print(f"\n🔧 REAL-WORLD APPLICABILITY:")
    print(f"   ✓ No aggressive augmentation - generalizes to real data")
    print(f"   ✓ Conservative hyperparameters - stable predictions")
    print(f"   ✓ Early stopping - prevents overfitting")
    print(f"   ✓ Ensemble - robust to individual model variance")
    
    print(f"\n📁 SAVED ARTIFACTS:")
    print(f"   Models:")
    print(f"   • results/models/ensemble_model_1.pt (Standard)")
    print(f"   • results/models/ensemble_model_2.pt (Large)")
    print(f"   • results/models/ensemble_model_3.pt (XLarge)")
    print(f"   Training histories:")
    print(f"   • results/training/model_1_history.json")
    print(f"   • results/training/model_2_history.json")
    print(f"   • results/training/model_3_history.json")
    
    print(f"\n▶️  NEXT STEPS:")
    print(f"   1. Run evaluate.py to generate visualizations")
    print(f"      python evaluate.py")
    print(f"   2. Analyze results against publication criteria")
    print(f"   3. Write research paper with methodology section")
    
    print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    main()