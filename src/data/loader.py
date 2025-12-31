"""
Data loading for 479-feature EEG dataset with 33,435 samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

class EEGDataLoader:
    """Load and prepare 479-feature EEG data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        
    def load(self) -> pd.DataFrame:
        """Load CSV - LARGE FILE HANDLING"""
        print(f"Loading data from: {self.csv_path}")
        print("⚠️  This is a large file (~500MB+). Please wait...")
        
        try:
            # Load only numeric data to save memory
            self.df = pd.read_csv(self.csv_path, dtype=np.float32)
            print(f"✓ Loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Samples: {self.df.shape}")
            print(f"  Features: {self.df.shape}")
            return self.df
        except MemoryError:
            print("❌ File too large for memory!")
            print("Try reading in chunks or using chunked processing")
            raise
    
    def get_feature_columns(self) -> list:
        """Get all feature columns (A to XQ)"""
        # Exclude BIS and Subject
        exclude = ['BIS', 'Subject']
        features = [col for col in self.df.columns if col not in exclude]
        return features
    
    def get_bis_column(self) -> str:
        """Get BIS column name"""
        if 'BIS' in self.df.columns:
            return 'BIS'
        raise ValueError("BIS column not found!")
    
    def get_subject_column(self) -> str:
        """Get Subject column name"""
        if 'Subject' in self.df.columns:
            return 'Subject'
        return None
    
    def get_stats(self) -> dict:
        """Get dataset statistics"""
        feature_cols = self.get_feature_columns()
        bis_col = self.get_bis_column()
        
        return {
            'num_features': len(feature_cols),
            'num_samples': len(self.df),
            'feature_range': (self.df[feature_cols].min().min(), 
                            self.df[feature_cols].max().max()),
            'bis_range': (self.df[bis_col].min(), self.df[bis_col].max()),
        }
    
    def prepare_data(self, test_size: float = 0.15,
                    val_size: float = 0.15,
                    random_state: int = 42) -> Tuple:
        """
        Prepare data with stratified splitting for large dataset
        
        ⚠️  Returns indices, not actual arrays (memory efficient)
        """
        n_samples = len(self.df)
        n_test = int(n_samples * test_size)
        n_val = int((n_samples - n_test) * val_size)
        
        # Shuffle
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # Split
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        
        print(f"Data split:")
        print(f"  Train: {len(train_idx)} ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"  Val:   {len(val_idx)} ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} ({len(test_idx)/n_samples*100:.1f}%)")
        
        return train_idx, val_idx, test_idx
    
    def get_feature_subset(self, indices: np.ndarray, 
                          max_features: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features and labels for given indices
        
        Optionally select top features to reduce dimensionality
        """
        feature_cols = self.get_feature_columns()
        bis_col = self.get_bis_column()
        
        # Get data
        X = self.df.iloc[indices][feature_cols].values.astype(np.float32)
        y = self.df.iloc[indices][bis_col].values.astype(np.float32)
        
        # Optionally reduce features
        if max_features and len(feature_cols) > max_features:
            print(f"⚠️  Reducing from {len(feature_cols)} to {max_features} features")
            # Select top variance features
            var = np.var(X, axis=0)
            top_idx = np.argsort(var)[-max_features:]
            X = X[:, top_idx]
        
        return X, y
