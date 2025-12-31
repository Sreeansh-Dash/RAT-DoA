#!/usr/bin/env python3
"""
IMPROVED Training script for RAT-DoA model
With multiple improvements to boost performance
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
    print("RAT-DoA IMPROVED Training")
    print("="*80)
    
    # ===== Configuration =====
    CONFIG = {
        'input_dim': 479,
        'embedding_dim': 512,  # INCREASED from 256
        'resnet_dims': [1024, 2048],  # INCREASED from [512, 1024]
        'attention_heads': 8,
        'transformer_layers': 3,
        'dropout': 0.15,  # REDUCED from 0.2
        
        'batch_size': 16,
        'learning_rate': 1e-3,  # INCREASED from 1e-4
        'weight_decay': 1e-5,
        'epochs': 200,  # INCREASED from 50
        'early_stopping_patience': 30,  # INCREASED from 15
        
        'use_feature_selection': True,
        'n_selected_features': 300,  # Select top 300 out of 479
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # ===== Load Data =====
    print("\n1. Loading dataset...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    
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
    
    # Normalize features
    X_train, train_params = prep.normalize_features(X_train)
    X_val, _ = prep.normalize_features(X_val, train_params['means'], train_params['stds'])
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    
    # Normalize BIS
    y_train, bis_params = prep.normalize_bis(y_train)
    y_val, _ = prep.normalize_bis(y_val, bis_params['mean'], bis_params['std'])
    y_test, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    # ===== Remove Outliers =====
    print("\n5. Removing outliers...")
    X_train, y_train = prep.remove_outliers(X_train, y_train, n_std=3.0)
    
    # ===== Feature Selection (OPTIONAL BUT RECOMMENDED) =====
    if CONFIG['use_feature_selection']:
        print(f"\n6. Selecting top {CONFIG['n_selected_features']} features...")
        X_train, y_train, top_indices = prep.select_top_features(
            X_train, y_train, 
            n_features=CONFIG['n_selected_features']
        )
        X_val, _, _ = prep.select_top_features(
            X_val, y_val, 
            n_features=CONFIG['n_selected_features']
        )
        X_test, _, _ = prep.select_top_features(
            X_test, y_test, 
            n_features=CONFIG['n_selected_features']
        )
        
        # Update config
        CONFIG['input_dim'] = CONFIG['n_selected_features']
    
    # ===== Create DataLoaders =====
    print("\n7. Creating DataLoaders...")
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False
    )
    
    # ===== Create Model =====
    print("\n8. Creating improved RAT-DoA model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RAT_DoA(
        input_dim=CONFIG['input_dim'],
        embedding_dim=CONFIG['embedding_dim'],
        resnet_dims=CONFIG['resnet_dims'],
        attention_heads=CONFIG['attention_heads'],
        transformer_layers=CONFIG['transformer_layers'],
        dropout=CONFIG['dropout']
    )
    print(f"   Device: {device}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # ===== Train =====
    print("\n9. Starting improved training...")
    print(f"   Max epochs: {CONFIG['epochs']}")
    print(f"   Early stopping patience: {CONFIG['early_stopping_patience']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    
    trainer = RAT_Trainer(
        model, 
        device=device,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=CONFIG['epochs'],
        patience=CONFIG['early_stopping_patience']
    )
    
    # ===== Test =====
    print("\n10. Testing...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    test_mae, test_preds, test_labels = trainer.validate(test_loader)
    
    # Denormalize
    test_preds_orig = prep.denormalize_bis(np.array(test_preds), bis_params)
    test_labels_orig = prep.denormalize_bis(np.array(test_labels), bis_params)
    test_mae_orig = np.mean(np.abs(test_preds_orig - test_labels_orig))
    
    print(f"\n   Test MAE (normalized): {test_mae:.4f}")
    print(f"   Test MAE (original 0-100): {test_mae_orig:.4f}")
    
    # ===== Comparison =====
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"Previous MAE:  10.91")
    print(f"Current MAE:   {test_mae_orig:.4f}")
    if test_mae_orig < 10.91:
        improvement = ((10.91 - test_mae_orig) / 10.91) * 100
        print(f"Improvement:   {improvement:.1f}% ✓")
    else:
        print(f"Improvement:   Worse ✗ (need more tuning)")
    
    print("\n" + "="*80)
    print("Key Improvements Made:")
    print("="*80)
    print("✓ Increased model capacity (embedding: 256→512, dims: 512/1024→1024/2048)")
    print("✓ Increased learning rate (1e-4 → 1e-3)")
    print("✓ Increased epochs (50 → 200)")
    print("✓ Increased early stopping patience (15 → 30)")
    print("✓ Reduced dropout (0.2 → 0.15) for better learning")
    print(f"✓ Feature selection ({CONFIG['n_selected_features']} best features)")
    print("✓ Added pin_memory for faster GPU transfer")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()