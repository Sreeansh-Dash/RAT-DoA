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
    model_path = Path('results/models/best_model.pt')

    model = None
    if model_path.exists():
        # Load checkpoint to CPU and try to infer model architecture
        ckpt = torch.load(model_path, map_location='cpu')
        try:
            # If checkpoint is a dict with 'state_dict' key, extract it
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        except Exception:
            state_dict = ckpt

        inferred = {}
        # Try to infer input and embedding dims from feature_embedding weights
        if isinstance(state_dict, dict) and 'feature_embedding.0.weight' in state_dict:
            w0 = state_dict['feature_embedding.0.weight']
            # w0 shape: (embedding_dim*2, input_dim)
            inferred['input_dim'] = int(w0.shape[1])
            inferred['embedding_dim'] = int(w0.shape[0] // 2)

        # Try to infer resnet dims
        if isinstance(state_dict, dict) and 'resnet_blocks.0.linear1.weight' in state_dict:
            r0 = state_dict['resnet_blocks.0.linear1.weight']
            inferred['resnet_0'] = int(r0.shape[0])
        if isinstance(state_dict, dict) and 'resnet_blocks.1.linear1.weight' in state_dict:
            r1 = state_dict['resnet_blocks.1.linear1.weight']
            inferred['resnet_1'] = int(r1.shape[0])

        # Build model using inferred values when possible, else fall back to dataset shape
        try:
            if 'input_dim' in inferred and 'embedding_dim' in inferred:
                kwargs = {
                    'input_dim': inferred['input_dim'],
                    'embedding_dim': inferred['embedding_dim']
                }
                if 'resnet_0' in inferred and 'resnet_1' in inferred:
                    kwargs['resnet_dims'] = [inferred['resnet_0'], inferred['resnet_1']]
                model = RAT_DoA(**kwargs)
                print(f"   ✓ Inferred model from checkpoint: input_dim={kwargs['input_dim']}, embedding_dim={kwargs['embedding_dim']}, resnet_dims={kwargs.get('resnet_dims')}")
            else:
                # fallback
                model = RAT_DoA(input_dim=X_test.shape[1])
                print(f"   ⚠️  Could not fully infer architecture; using input_dim={X_test.shape[1]}")
        except Exception as e:
            print(f"   ⚠️  Failed to construct inferred model: {e}")
            model = RAT_DoA(input_dim=X_test.shape[1])

        # Attempt strict load first, then non-strict partial load
        try:
            model.load_state_dict(state_dict)
            print(f"   ✓ Loaded full state dict from {model_path}")
        except RuntimeError as e:
            print(f"   ⚠️  Full load failed: {e}")
            print("   Attempting partial (non-strict) load and skipping mismatched layers...")
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✓ Partially loaded state dict from {model_path}")
    else:
        print(f"   ⚠️  Warning: {model_path} not found!")
        print(f"   Using untrained model for demonstration")
        model = RAT_DoA(input_dim=X_test.shape[1])

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