#!/usr/bin/env python3

"""
PHASE 2: ENSEMBLE TRAINING FOR BETTER RESULTS
===============================================
Train 3 RAT models with different seeds + better hyperparameters
Expected: MAE 2.5-3.5 (from 11.35)
Time: 4-5 hours on CPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.data.preprocessor import LargeFeaturePreprocessor
from src.models.rat_model import RAT_DoA, count_parameters
from src.training.trainer import RAT_Trainer

class AugmentedDataLoader(DataLoader):
    """DataLoader with on-the-fly augmentation"""
    def __init__(self, *args, augment=False, noise_std=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.noise_std = noise_std
    
    def __iter__(self):
        for batch_x, batch_y in super().__iter__():
            if self.augment:
                noise = torch.randn_like(batch_x) * self.noise_std
                batch_x = batch_x + noise
            yield batch_x, batch_y

def create_larger_model(input_dim):
    """Larger model for better capacity"""
    return RAT_DoA(
        input_dim=input_dim,
        embedding_dim=256,      # Larger: 128→256
        resnet_dims=[512, 512], # Larger: [256,256]→[512,512]
        attention_heads=8,      # More heads
        transformer_layers=2,   # More layers
        dropout=0.1,            # Lower dropout
    )

def main():
    print("\n" + "="*90)
    print("🚀 PHASE 2: ENSEMBLE TRAINING FOR BETTER RESULTS")
    print("="*90)
    print(f"Goal: MAE < 3.0")
    print(f"Time: ~4-5 hours")
    print(f"Strategy: 3x RAT models (different seeds) + ensemble")
    print("="*90)
    
    # ===== Load Data =====
    print("\n1️⃣  Loading data...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    stats = loader.get_stats()
    print(f"   ✓ Features: {stats['num_features']}")
    print(f"   ✓ Samples: {stats['num_samples']:,}")
    
    # ===== Split Data =====
    print("\n2️⃣  Splitting data...")
    train_idx, val_idx, test_idx = loader.prepare_data()
    
    # ===== Load Features =====
    print("\n3️⃣  Loading features...")
    X_train, y_train = loader.get_feature_subset(train_idx)
    X_val, y_val = loader.get_feature_subset(val_idx)
    X_test, y_test = loader.get_feature_subset(test_idx)
    print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ===== Preprocess =====
    print("\n4️⃣  Preprocessing...")
    prep = LargeFeaturePreprocessor()
    
    X_train, train_params = prep.normalize_features(X_train)
    X_val, _ = prep.normalize_features(X_val, train_params['means'], train_params['stds'])
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    
    y_train, bis_params = prep.normalize_bis(y_train)
    y_val, _ = prep.normalize_bis(y_val, bis_params['mean'], bis_params['std'])
    y_test, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    X_train_orig_size = X_train.shape[0]
    X_train, y_train = prep.remove_outliers(X_train, y_train)
    print(f"   ✓ Outliers removed: {X_train_orig_size - X_train.shape[0]}")
    
    # ===== Create DataLoaders =====
    print("\n5️⃣  Creating DataLoaders...")
    batch_size = 32  # Larger batch size
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Augmentation: σ=0.02 (stronger)")
    
    # ===== ENSEMBLE: Train 3 Models =====
    print("\n6️⃣  ENSEMBLE TRAINING: 3 Models with different seeds")
    print("   (Each model ~90-120 min on CPU)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ✓ Device: {device}")
    
    ensemble_predictions = []
    ensemble_models = []
    seeds = [42, 123, 456]
    
    for model_idx, seed in enumerate(seeds, 1):
        print(f"\n   {'='*70}")
        print(f"   📦 MODEL {model_idx}/3 (seed={seed})")
        print(f"   {'='*70}")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create larger model
        input_dim = X_train.shape[1]
        model = create_larger_model(input_dim)
        print(f"      • Input: {input_dim}, Params: {count_parameters(model):,}")
        print(f"      • Architecture: embedding=256, resnet=[512,512], attn_heads=8")
        
        # DataLoaders with augmentation
        train_loader = AugmentedDataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            augment=True,
            noise_std=0.02
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train
        trainer = RAT_Trainer(model, device=device)
        trainer.learning_rate = 5e-3      # Very aggressive: 1e-3 → 5e-3
        trainer.weight_decay = 1e-5
        trainer.criterion = nn.HuberLoss(delta=1.0)
        
        print(f"      • LR: 5e-3 (very aggressive)")
        print(f"      • Loss: HuberLoss")
        print(f"      • Epochs: 150, Patience: 25")
        print(f"      ⏳ Training...")
        
        history = trainer.train(
            train_loader, 
            val_loader, 
            epochs=150,
            patience=25
        )
        
        # Get validation predictions
        val_loader_eval = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_mae, val_preds, val_labels = trainer.validate(val_loader_eval)
        
        print(f"      ✓ Val MAE (norm): {val_mae:.4f}")
        
        # Save model
        ensemble_models.append(trainer.model)
        torch.save(trainer.model.state_dict(), f'results/models/ensemble_model_{model_idx}.pt')
        print(f"      ✓ Saved: results/models/ensemble_model_{model_idx}.pt")
    
    # ===== ENSEMBLE PREDICTION =====
    print(f"\n7️⃣  ENSEMBLE PREDICTION (averaging 3 models)")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    # Get predictions from all 3 models
    for model_idx, model in enumerate(ensemble_models, 1):
        model = model.to(device)
        model.eval()
        
        preds = []
        labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                preds.append(output.cpu().numpy())
                labels.append(batch_y.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0).flatten()
        labels = np.concatenate(labels, axis=0).flatten()
        
        all_predictions.append(preds)
        if model_idx == 1:
            all_labels = labels
        
        print(f"   Model {model_idx}: Got {len(preds)} predictions")
    
    # Ensemble: Average predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    print(f"   ✓ Ensemble: Average of 3 models")
    
    # Denormalize
    ensemble_pred_orig = prep.denormalize_bis(ensemble_pred, bis_params)
    labels_orig = prep.denormalize_bis(all_labels, bis_params)
    
    # Metrics
    ensemble_mae = np.mean(np.abs(ensemble_pred_orig - labels_orig))
    
    from scipy.stats import pearsonr
    pearson_r, _ = pearsonr(ensemble_pred_orig, labels_orig)
    
    ss_res = np.sum((labels_orig - ensemble_pred_orig) ** 2)
    ss_tot = np.sum((labels_orig - np.mean(labels_orig)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # ===== RESULTS =====
    print("\n" + "="*90)
    print("✨ ENSEMBLE TRAINING COMPLETE!")
    print("="*90)
    
    print(f"\n📊 TEST SET RESULTS:")
    print(f"   MAE (0-100 scale):    {ensemble_mae:.4f}")
    print(f"   Pearson r:            {pearson_r:.4f}")
    print(f"   R² Score:             {r2_score:.4f}")
    
    print(f"\n📁 Models saved:")
    print(f"   - results/models/ensemble_model_1.pt")
    print(f"   - results/models/ensemble_model_2.pt")
    print(f"   - results/models/ensemble_model_3.pt")
    
    # Save ensemble results
    ensemble_results = {
        'mae': float(ensemble_mae),
        'pearson_r': float(pearson_r),
        'r2_score': float(r2_score),
        'predictions': ensemble_pred_orig.tolist(),
        'labels': labels_orig.tolist(),
        'seeds': seeds,
        'model_type': 'RAT-DoA (ensemble of 3)',
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path('results/ensemble_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(ensemble_results, f, indent=2)
    print(f"   ✓ Ensemble results: results/ensemble_results.json")
    
    # ===== VERDICT =====
    print(f"\n🎯 RESULTS ASSESSMENT:")
    if ensemble_mae < 2.5:
        print(f"   🎉 EXCELLENT! MAE < 2.5 - Publication quality!")
    elif ensemble_mae < 3.0:
        print(f"   ✅ GREAT! MAE < 3.0 - Target achieved!")
    elif ensemble_mae < 4.0:
        print(f"   👍 GOOD! MAE < 4.0 - Acceptable performance")
        print(f"   💡 Next: Try ensemble with TCN model (different architecture)")
    else:
        print(f"   ⚠️  MAE still high. Try:")
        print(f"      - Even larger model (embedding=512, resnet=[1024,1024])")
        print(f"      - Mix TCN + RAT ensemble")
        print(f"      - Feature engineering")
    
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    main()