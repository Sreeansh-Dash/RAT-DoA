"""
Preprocessing for 479-feature dataset
Focus on: normalization, dimensionality reduction, outlier handling
"""

import numpy as np
from typing import Tuple, Dict

class LargeFeaturePreprocessor:
    """Handle 479-feature preprocessing"""
    
    @staticmethod
    def normalize_features(X: np.ndarray,
                          means: np.ndarray = None,
                          stds: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Z-score normalization (critical for large feature sets!)
        
        Args:
            X: [N, 479] feature matrix
            means: Pre-computed means (for test set)
            stds: Pre-computed stds (for test set)
        """
        if means is None:
            means = X.mean(axis=0)
        if stds is None:
            stds = X.std(axis=0) + 1e-8
        
        X_norm = (X - means) / stds
        
        return X_norm, {'means': means, 'stds': stds}
    
    @staticmethod
    def normalize_bis(y: np.ndarray,
                     mean: float = None,
                     std: float = None) -> Tuple[np.ndarray, Dict]:
        """Normalize BIS (0-100) to ~(0, 1)"""
        if mean is None:
            mean = y.mean()
        if std is None:
            std = y.std() + 1e-8
        
        y_norm = (y - mean) / std
        return y_norm, {'mean': mean, 'std': std}
    
    @staticmethod
    def denormalize_bis(y_norm: np.ndarray, params: Dict) -> np.ndarray:
        """Convert back to 0-100 range"""
        return y_norm * params['std'] + params['mean']
    
    @staticmethod
    def select_top_features(X: np.ndarray, y: np.ndarray,
                           n_features: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select top N features by variance (reduce 479 → 256)
        
        This is important: 479 features might have redundancy/noise
        """
        # Calculate variance
        var = np.var(X, axis=0)
        
        # Select top N
        top_indices = np.argsort(var)[-n_features:]
        X_reduced = X[:, top_indices]
        
        print(f"Selected top {n_features} features out of {X.shape}")
        
        return X_reduced, y, top_indices
    
    @staticmethod
    def remove_outliers(X: np.ndarray, y: np.ndarray,
                       n_std: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Remove extreme outliers (beyond 3 std)"""
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        
        outlier_mask = np.abs((X - means) / (stds + 1e-8)) > n_std
        outlier_samples = outlier_mask.any(axis=1)
        
        X_clean = X[~outlier_samples]
        y_clean = y[~outlier_samples]
        
        removed = outlier_samples.sum()
        if removed > 0:
            print(f"Removed {removed} outlier samples ({removed/len(y)*100:.2f}%)")
        
        return X_clean, y_clean
