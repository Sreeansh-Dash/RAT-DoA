"""
Model Evaluation and Analysis
Comprehensive evaluation metrics and visualization
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation
    Computes metrics and creates visualizations
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, 
                 data_loader: DataLoader,
                 denorm_func=None,
                 denorm_params: Dict = None) -> Dict:
        """
        Comprehensive evaluation
        
        Args:
            data_loader: Test data loader
            denorm_func: Function to denormalize predictions
            denorm_params: Parameters for denormalization
            
        Returns:
            metrics: Dictionary of all metrics
        """
        predictions, labels = self._get_predictions(data_loader)
        
        # Denormalize if function provided
        if denorm_func and denorm_params:
            predictions = denorm_func(predictions, denorm_params)
            labels = denorm_func(labels, denorm_params)
        
        metrics = self._compute_metrics(predictions, labels)
        return metrics
    
    def _get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions on dataset"""
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)
                
                all_predictions.extend(predictions.cpu().numpy().flatten().tolist())
                all_labels.extend(batch_y.cpu().numpy().flatten().tolist())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _compute_metrics(self, 
                        predictions: np.ndarray,
                        labels: np.ndarray) -> Dict:
        """Compute all evaluation metrics"""
        
        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(labels, predictions)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(labels, predictions)
        
        # Mean error (bias)
        mean_error = np.mean(predictions - labels)
        
        # Standard deviation of errors
        std_error = np.std(predictions - labels)
        
        # Median absolute error
        median_ae = np.median(np.abs(predictions - labels))
        
        # 95% confidence interval
        ae = np.abs(predictions - labels)
        ci_95 = np.percentile(ae, 95)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'median_ae': float(median_ae),
            'ci_95': float(ci_95),
            'num_samples': len(labels)
        }
        
        return metrics
    
    def plot_results(self,
                    predictions: np.ndarray,
                    labels: np.ndarray,
                    save_dir: str = 'results/figures',
                    prefix: str = 'test'):
        """
        Create evaluation plots
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            save_dir: Directory to save figures
            prefix: Prefix for saved figures
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{prefix.upper()} SET EVALUATION', fontsize=16, fontweight='bold')
        
        # 1. Prediction vs Ground Truth Scatter
        ax = axes[0, 0]
        ax.scatter(labels, predictions, alpha=0.5, s=20)
        ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Predictions', fontsize=12)
        ax.set_title('Predictions vs Ground Truth', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax = axes[0, 1]
        residuals = predictions - labels
        ax.scatter(labels, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Residuals (Pred - True)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax = axes[0, 2]
        ae = np.abs(predictions - labels)
        ax.hist(ae, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(ae), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(ae):.2f}')
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Bland-Altman Plot
        ax = axes[1, 0]
        mean_vals = (predictions + labels) / 2
        diffs = predictions - labels
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        ax.scatter(mean_vals, diffs, alpha=0.5, s=20)
        ax.axhline(y=mean_diff, color='r', linestyle='-', lw=2, label='Mean difference')
        ax.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle='--', lw=1)
        ax.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle='--', lw=1)
        ax.set_xlabel('Mean of Predictions and Truth', fontsize=12)
        ax.set_ylabel('Difference', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Prediction Error vs BIS Level
        ax = axes[1, 1]
        ae_by_bin = []
        bins = np.linspace(labels.min(), labels.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        for i in range(len(bins) - 1):
            mask = (labels >= bins[i]) & (labels < bins[i+1])
            if mask.sum() > 0:
                ae_by_bin.append(np.mean(ae[mask]))
            else:
                ae_by_bin.append(np.nan)
        ax.plot(bin_centers, ae_by_bin, 'o-', lw=2, markersize=8, color='steelblue')
        ax.set_xlabel('BIS Level', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title('MAE by BIS Level', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 6. Metrics Summary (text)
        ax = axes[1, 2]
        ax.axis('off')
        metrics = self._compute_metrics(predictions, labels)
        text_str = f"""
EVALUATION METRICS

MAE:         {metrics['mae']:.4f}
RMSE:        {metrics['rmse']:.4f}
MSE:         {metrics['mse']:.4f}
R² Score:    {metrics['r2_score']:.4f}

Pearson r:   {metrics['pearson_r']:.4f}
Spearman r:  {metrics['spearman_r']:.4f}

Mean Error:  {metrics['mean_error']:.4f}
Std Error:   {metrics['std_error']:.4f}
Median AE:   {metrics['median_ae']:.4f}
95% CI:      {metrics['ci_95']:.4f}

Samples:     {metrics['num_samples']}
        """
        ax.text(0.1, 0.95, text_str, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        save_path = save_dir / f'{prefix}_evaluation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved evaluation plot to {save_path}")
        
        plt.close()
    
    def save_metrics(self,
                    metrics: Dict,
                    save_path: str = 'results/metrics.json'):
        """Save metrics to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved metrics to {save_path}")
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted way"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"MAE:              {metrics['mae']:.6f}")
        print(f"RMSE:             {metrics['rmse']:.6f}")
        print(f"MSE:              {metrics['mse']:.6f}")
        print(f"R² Score:         {metrics['r2_score']:.6f}")
        print(f"\nPearson r:        {metrics['pearson_r']:.6f} (p={metrics['pearson_p']:.2e})")
        print(f"Spearman r:       {metrics['spearman_r']:.6f} (p={metrics['spearman_p']:.2e})")
        print(f"\nMean Error:       {metrics['mean_error']:.6f}")
        print(f"Std Error:        {metrics['std_error']:.6f}")
        print(f"Median AE:        {metrics['median_ae']:.6f}")
        print(f"95% CI:           {metrics['ci_95']:.6f}")
        print(f"\nSamples:          {metrics['num_samples']}")
        print("="*60 + "\n")