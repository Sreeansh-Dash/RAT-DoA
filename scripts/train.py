#!/usr/bin/env python3
"""
Training script for 479-feature EEG data
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.data.preprocessor import LargeFeaturePreprocessor
from src.models.rat_model import RAT_DoA, count_parameters
from src.training.trainer import RAT_Trainer

def main():
    print("RAT-DoA Training (479-Feature Version)")
    print("="*80)
    
    # ===== Load Data =====
    print("\n1. Loading 33,435 samples × 479 features...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    
    stats = loader.get_stats()
    print(f"   Features: {stats['num_features']}")
    print(f"   Samples: {stats['num_samples']:,}")
    
    # ===== Split Data =====
    print("\n2. Splitting data...")
    train_idx, val_idx, test_idx = loader.prepare_data()
    
    # ===== Load Features =====
    print("\n3. Loading features...")
    X_train, y_train = loader.get_feature_subset(train_idx)
    X_val, y_val = loader.get_feature_subset(val_idx)
    X_test, y_test = loader.get_feature_subset(test_idx)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # ===== Preprocess =====
    print("\n4. Preprocessing...")
    prep = LargeFeaturePreprocessor()
    
    X_train, train_params = prep.normalize_features(X_train)
    X_val, _ = prep.normalize_features(X_val, train_params['means'], train_params['stds'])
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    
    y_train, bis_params = prep.normalize_bis(y_train)
    y_val, _ = prep.normalize_bis(y_val, bis_params['mean'], bis_params['std'])
    y_test, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    # ===== Remove Outliers =====
    print("\n5. Removing outliers...")
    X_train, y_train = prep.remove_outliers(X_train, y_train)
    
    # ===== Create DataLoaders =====
    print("\n6. Creating DataLoaders...")
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    # IMPORTANT: Smaller batch size for large feature set
    batch_size = 16  # Reduced from 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ===== Create Model =====
    print("\n7. Creating RAT-DoA model...")
    input_dim = X_train.shape[1]
    model = RAT_DoA(input_dim=input_dim)
    print(f"   Input dim: {input_dim}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # ===== Train =====
    print("\n8. Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    trainer = RAT_Trainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=50, patience=15)  # Reduced epochs
    
    # ===== Test =====
    print("\n9. Testing...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    test_mae, test_preds, test_labels = trainer.validate(test_loader)
    
    # Denormalize
    test_preds_orig = prep.denormalize_bis(np.array(test_preds), bis_params)
    test_labels_orig = prep.denormalize_bis(np.array(test_labels), bis_params)
    test_mae_orig = np.mean(np.abs(test_preds_orig - test_labels_orig))
    
    print(f"\n   Test MAE (normalized): {test_mae:.4f}")
    print(f"   Test MAE (original 0-100): {test_mae_orig:.4f}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
