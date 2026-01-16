#!/usr/bin/env python3

"""
ENSEMBLE EVALUATION SCRIPT
==========================
Comprehensive evaluation of ensemble model predictions
Generates detailed metrics and visualizations
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import EEGDataLoader
from src.data.preprocessor import LargeFeaturePreprocessor
from src.models.rat_model import RAT_DoA, count_parameters

def create_larger_model(input_dim):
    """Larger model for better capacity"""
    return RAT_DoA(
        input_dim=input_dim,
        embedding_dim=256,
        resnet_dims=[512, 512],
        attention_heads=8,
        transformer_layers=2,
        dropout=0.1,
    )

def compute_metrics(predictions, labels):
    """Compute comprehensive metrics"""
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    mse = np.mean((predictions - labels) ** 2)
    
    pearson_r, pearson_p = pearsonr(predictions, labels)
    spearman_r, spearman_p = spearmanr(predictions, labels)
    
    # R² Score
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((labels - predictions) / (labels + 1e-6))) * 100
    
    # Error statistics
    errors = np.abs(predictions - labels)
    mean_error = np.mean(predictions - labels)
    std_error = np.std(predictions - labels)
    median_ae = np.median(errors)
    
    # Confidence interval (95%)
    ci_95 = 1.96 * np.std(errors) / np.sqrt(len(errors))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r2_score': r2_score,
        'mape': mape,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_ae': median_ae,
        'ci_95': ci_95,
    }

def plot_predictions_vs_truth(predictions, labels, save_path):
    """Plot predictions vs ground truth"""
    plt.figure(figsize=(8, 6))
    plt.scatter(labels, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    plt.xlabel('Ground Truth BIS', fontsize=12)
    plt.ylabel('Predicted BIS', fontsize=12)
    plt.title('Ensemble: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_residuals(predictions, labels, save_path):
    """Plot residuals and error distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = predictions - labels
    
    # Residual plot
    axes[0].scatter(labels, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Ground Truth BIS', fontsize=11)
    axes[0].set_ylabel('Residuals (Pred - Truth)', fontsize=11)
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = np.abs(residuals)
    axes[1].hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[1].axvline(np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):.2f}')
    axes[1].set_xlabel('Absolute Error', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_bland_altman(predictions, labels, save_path):
    """Bland-Altman plot for agreement"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_vals = (predictions + labels) / 2
    diff_vals = predictions - labels
    
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20)
    ax.axhline(y=mean_diff, color='r', linestyle='-', lw=2, label=f'Mean difference: {mean_diff:.2f}')
    ax.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle='--', lw=1.5, label=f'±1.96 SD: {1.96*std_diff:.2f}')
    ax.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle='--', lw=1.5)
    
    ax.set_xlabel('Mean of Predictions and Truth', fontsize=12)
    ax.set_ylabel('Difference (Pred - Truth)', fontsize=12)
    ax.set_title('Bland-Altman Plot (Agreement Analysis)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_mae_by_bis_level(predictions, labels, save_path, n_bins=10):
    """MAE breakdown by BIS level"""
    # Bin by BIS levels
    bis_bins = np.linspace(labels.min(), labels.max(), n_bins + 1)
    mae_by_level = []
    bin_centers = []
    
    for i in range(len(bis_bins) - 1):
        mask = (labels >= bis_bins[i]) & (labels < bis_bins[i+1])
        if mask.sum() > 0:
            mae = np.mean(np.abs(predictions[mask] - labels[mask]))
            mae_by_level.append(mae)
            bin_centers.append((bis_bins[i] + bis_bins[i+1]) / 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, mae_by_level, 'o-', markersize=8, linewidth=2, color='steelblue')
    plt.xlabel('BIS Level', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title('MAE by BIS Level (Performance across range)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def plot_summary(predictions, labels, metrics, save_path):
    """Summary dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Predictions vs Truth
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(labels, predictions, alpha=0.4, s=15, color='steelblue')
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax1.set_xlabel('Ground Truth', fontsize=10)
    ax1.set_ylabel('Predictions', fontsize=10)
    ax1.set_title('Predictions vs Ground Truth', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""
METRICS SUMMARY
    
MAE: {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}
R²: {metrics['r2_score']:.4f}

Pearson r: {metrics['pearson_r']:.4f}
Spearman r: {metrics['spearman_r']:.4f}

MAPE: {metrics['mape']:.2f}%
Median AE: {metrics['median_ae']:.4f}
95% CI: ±{metrics['ci_95']:.4f}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='center')
    
    # 3. Residuals
    ax3 = fig.add_subplot(gs[1, 0])
    residuals = predictions - labels
    ax3.scatter(labels, residuals, alpha=0.4, s=15, color='steelblue')
    ax3.axhline(y=0, color='r', linestyle='--', lw=1.5)
    ax3.set_xlabel('Ground Truth', fontsize=10)
    ax3.set_ylabel('Residuals', fontsize=10)
    ax3.set_title('Residual Plot', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error histogram
    ax4 = fig.add_subplot(gs[1, 1])
    errors = np.abs(residuals)
    ax4.hist(errors, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(errors), color='r', linestyle='--', lw=1.5, label=f'Mean: {np.mean(errors):.2f}')
    ax4.axvline(np.median(errors), color='g', linestyle='--', lw=1.5, label=f'Median: {np.median(errors):.2f}')
    ax4.set_xlabel('Absolute Error', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Bland-Altman
    ax5 = fig.add_subplot(gs[1, 2])
    mean_vals = (predictions + labels) / 2
    diff_vals = residuals
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    ax5.scatter(mean_vals, diff_vals, alpha=0.4, s=15, color='steelblue')
    ax5.axhline(y=mean_diff, color='r', linestyle='-', lw=1.5)
    ax5.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle='--', lw=1)
    ax5.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle='--', lw=1)
    ax5.set_xlabel('Mean (Pred+Truth)/2', fontsize=10)
    ax5.set_ylabel('Difference', fontsize=10)
    ax5.set_title('Bland-Altman Plot', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. MAE by BIS level
    ax6 = fig.add_subplot(gs[2, :])
    bis_bins = np.linspace(labels.min(), labels.max(), 10)
    mae_by_level = []
    bin_centers = []
    for i in range(len(bis_bins) - 1):
        mask = (labels >= bis_bins[i]) & (labels < bis_bins[i+1])
        if mask.sum() > 0:
            mae = np.mean(np.abs(predictions[mask] - labels[mask]))
            mae_by_level.append(mae)
            bin_centers.append((bis_bins[i] + bis_bins[i+1]) / 2)
    ax6.plot(bin_centers, mae_by_level, 'o-', markersize=8, linewidth=2, color='steelblue')
    ax6.set_xlabel('BIS Level', fontsize=10)
    ax6.set_ylabel('Mean Absolute Error', fontsize=10)
    ax6.set_title('Performance Across BIS Levels', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('ENSEMBLE MODEL EVALUATION SUMMARY', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {save_path}")

def main():
    print("\n" + "="*90)
    print("📊 ENSEMBLE MODEL EVALUATION")
    print("="*90)
    
    # ===== Load Data =====
    print("\n1️⃣  Loading data...")
    loader = EEGDataLoader('data/raw/EEG_BIS_Segments.csv')
    df = loader.load()
    
    # ===== Split Data =====
    print("2️⃣  Splitting data...")
    train_idx, val_idx, test_idx = loader.prepare_data()
    
    # ===== Load Features =====
    print("3️⃣  Loading test features...")
    X_test, y_test = loader.get_feature_subset(test_idx)
    X_train, y_train = loader.get_feature_subset(train_idx)
    
    # ===== Preprocess =====
    print("4️⃣  Preprocessing...")
    prep = LargeFeaturePreprocessor()
    
    X_train, train_params = prep.normalize_features(X_train)
    X_test, _ = prep.normalize_features(X_test, train_params['means'], train_params['stds'])
    
    y_train, bis_params = prep.normalize_bis(y_train)
    y_test_norm, _ = prep.normalize_bis(y_test, bis_params['mean'], bis_params['std'])
    
    # ===== Load Models =====
    print("\n5️⃣  Loading ensemble models (3x)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   ✓ Device: {device}")
    
    ensemble_models = []
    input_dim = X_test.shape[1]
    
    for model_idx in range(1, 4):
        model_path = Path(f'results/models/ensemble_model_{model_idx}.pt')
        
        if model_path.exists():
            model = create_larger_model(input_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            ensemble_models.append(model)
            print(f"   ✓ Model {model_idx} loaded: {model_path}")
        else:
            print(f"   ⚠️  Model {model_idx} not found: {model_path}")
    
    if len(ensemble_models) == 0:
        print("   ❌ No ensemble models found!")
        return
    
    # ===== Get Predictions =====
    print(f"\n6️⃣  Generating predictions from {len(ensemble_models)} models...")
    
    batch_size = 32
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test_norm))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = None
    
    for model_idx, model in enumerate(ensemble_models, 1):
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
        if all_labels is None:
            all_labels = labels
        
        print(f"   ✓ Model {model_idx}: {len(preds)} predictions")
    
    # Ensemble: Average
    ensemble_pred = np.mean(all_predictions, axis=0)
    print(f"   ✓ Ensemble: Average of {len(ensemble_models)} models")
    
    # ===== Denormalize =====
    print("\n7️⃣  Denormalizing predictions...")
    ensemble_pred_orig = prep.denormalize_bis(ensemble_pred, bis_params)
    labels_orig = prep.denormalize_bis(all_labels, bis_params)
    
    # ===== Compute Metrics =====
    print("8️⃣  Computing metrics...")
    metrics = compute_metrics(ensemble_pred_orig, labels_orig)
    
    # ===== Print Results =====
    print("\n" + "="*90)
    print("✨ ENSEMBLE EVALUATION RESULTS")
    print("="*90)
    
    print(f"\n📊 TEST SET METRICS (0-100 BIS scale):")
    print(f"   MAE:                {metrics['mae']:.4f}")
    print(f"   RMSE:               {metrics['rmse']:.4f}")
    print(f"   MSE:                {metrics['mse']:.4f}")
    print(f"   R² Score:           {metrics['r2_score']:.4f}")
    print(f"   Pearson r:          {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"   Spearman r:         {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"   MAPE:               {metrics['mape']:.2f}%")
    print(f"   Median AE:          {metrics['median_ae']:.4f}")
    print(f"   Mean Error:         {metrics['mean_error']:.4f}")
    print(f"   Std Error:          {metrics['std_error']:.4f}")
    print(f"   95% CI:             ±{metrics['ci_95']:.4f}")
    print(f"   Samples:            {len(labels_orig)}")
    
    # ===== Generate Plots =====
    print(f"\n9️⃣  Generating visualizations...")
    
    fig_dir = Path('results/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_predictions_vs_truth(
        ensemble_pred_orig, 
        labels_orig, 
        fig_dir / 'ensemble_predictions_vs_truth.png'
    )
    
    plot_residuals(
        ensemble_pred_orig, 
        labels_orig, 
        fig_dir / 'ensemble_residuals.png'
    )
    
    plot_bland_altman(
        ensemble_pred_orig, 
        labels_orig, 
        fig_dir / 'ensemble_bland_altman.png'
    )
    
    plot_mae_by_bis_level(
        ensemble_pred_orig, 
        labels_orig, 
        fig_dir / 'ensemble_mae_by_bis.png'
    )
    
    plot_summary(
        ensemble_pred_orig, 
        labels_orig, 
        metrics, 
        fig_dir / 'ensemble_evaluation_summary.png'
    )
    
    # ===== Save Results =====
    print(f"\n🔟 Saving results...")
    
    # Save metrics
    metrics_path = Path('results/ensemble_metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics: {metrics_path}")
    
    # Save predictions
    results = {
        'predictions': ensemble_pred_orig.tolist(),
        'labels': labels_orig.tolist(),
        'errors': (ensemble_pred_orig - labels_orig).tolist(),
        'metrics': metrics
    }
    
    results_path = Path('results/ensemble_predictions.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Predictions: {results_path}")
    
    # ===== Summary =====
    print("\n" + "="*90)
    print("🎯 EVALUATION COMPLETE!")
    print("="*90)
    
    print(f"\n📁 Results saved to:")
    print(f"   - Metrics: results/ensemble_metrics.json")
    print(f"   - Predictions: results/ensemble_predictions.json")
    print(f"   - Figures:")
    print(f"     • results/figures/ensemble_predictions_vs_truth.png")
    print(f"     • results/figures/ensemble_residuals.png")
    print(f"     • results/figures/ensemble_bland_altman.png")
    print(f"     • results/figures/ensemble_mae_by_bis.png")
    print(f"     • results/figures/ensemble_evaluation_summary.png")
    
    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if metrics['mae'] < 2.5:
        print(f"   🎉 EXCELLENT! MAE < 2.5 - Publication quality!")
    elif metrics['mae'] < 3.0:
        print(f"   ✅ GREAT! MAE < 3.0 - Target achieved!")
    elif metrics['mae'] < 4.0:
        print(f"   👍 GOOD! MAE < 4.0")
    else:
        print(f"   ⚠️  MAE still high - consider TCN ensemble or feature engineering")
    
    print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    main()