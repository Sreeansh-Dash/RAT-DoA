#!/usr/bin/env python3
"""
Evaluation script for RAT-DoA model
Comprehensive testing and analysis on test set
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
from src.training.evaluator import ModelEvaluator


def main():
    print("RAT-DoA Model Evaluation")
    print("="*80)
    
    # ===== Load Data =====
    print("\n1. Loading dataset...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    
    # ===== Split Data =====
    print("\n2. Splitting data...")
    train_idx, val_idx, test_idx = loader.prepare_data()
    
    # ===== Load Test Features =====
    print("\n3. Loading test features...")
    X_test, y_test = loader.get_feature_subset(test_idx)
    print(f"   Test set: {X_test.shape}")
    
    # ===== Preprocess =====
    print("\n4. Preprocessing...")
    prep = LargeFeaturePreprocessor()
    
    # Load training parameters from stored file (ideally)
    # For now, compute from training data
    X_train, y_train = loader.get_feature_subset(train_idx)
    X_train, train_params = prep.normalize_features(X_train)
    
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    y_train_norm, bis_params = prep.normalize_bis(y_train)
    y_test_norm, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    # Remove outliers from training
    X_train, y_train_norm = prep.remove_outliers(X_train, y_train_norm)
    
    # ===== Create DataLoader =====
    print("\n5. Creating test DataLoader...")
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test_norm))
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"   Batch size: {batch_size}")
    
    # ===== Load Model =====
    print("\n6. Loading trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X_test.shape[1]
    model = RAT_DoA(input_dim=input_dim)
    
    # Load best model weights
    model_path = Path('results/models/best_model.pt')
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   ✓ Loaded from {model_path}")
    else:
        print(f"   ⚠️  Warning: {model_path} not found!")
        print(f"   Using untrained model for demonstration")
    
    model = model.to(device)
    print(f"   Device: {device}")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # ===== Evaluation =====
    print("\n7. Running evaluation...")
    evaluator = ModelEvaluator(model, device=device)
    
    # Get predictions (normalized)
    predictions_norm, labels_norm = evaluator._get_predictions(test_loader)
    
    # Denormalize
    predictions = prep.denormalize_bis(predictions_norm, bis_params)
    labels = prep.denormalize_bis(labels_norm, bis_params)
    
    # Compute metrics
    metrics = evaluator._compute_metrics(predictions, labels)
    
    # ===== Print Results =====
    print("\n" + "="*80)
    print("TEST SET RESULTS (0-100 BIS Scale)")
    print("="*80)
    evaluator.print_metrics(metrics)
    
    # ===== Save Results =====
    print("\n8. Saving results...")
    
    # Save metrics
    evaluator.save_metrics(metrics, 'results/test_metrics.json')
    
    # Save predictions
    results = {
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'errors': (predictions - labels).tolist()
    }
    results_path = Path('results/test_predictions.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"✓ Saved predictions to {results_path}")
    
    # ===== Create Plots =====
    print("\n9. Creating evaluation plots...")
    evaluator.plot_results(predictions, labels, 'results/figures', prefix='test')
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Metrics: results/test_metrics.json")
    print(f"  - Predictions: results/test_predictions.json")
    print(f"  - Plots: results/figures/test_evaluation.png")


if __name__ == "__main__":
    main()