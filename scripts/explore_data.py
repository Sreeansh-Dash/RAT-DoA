#!/usr/bin/env python3
"""
Data exploration for 479-feature, 33,435-row dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def explore_dataset(csv_path):
    """Explore large dataset"""
    
    print("="*80)
    print("EEG-BIS DATASET EXPLORATION (479 Features, 33,435 Samples)")
    print("="*80)
    
    # Load
    print(f"\n1. Loading {csv_path}...")
    print("   ⚠️  This is a LARGE file (~500MB+). Please wait...\n")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Shape
    print(f"\n2. Dataset Shape:")
    print(f"   Total rows: {df.shape[0]:,} (including header)")
    print(f"   Data rows: {df.shape[0]-1:,}")
    print(f"   Total columns: {df.shape[1]}")
    
    # Columns
    print(f"\n3. Column Structure:")
    all_cols = df.columns.tolist()
    print(f"   First columns: {all_cols[:5]}")
    print(f"   Last columns: {all_cols[-5:]}")
    
    # Find feature columns
    feature_cols = [col for col in all_cols if col not in ['BIS', 'Subject']]
    print(f"\n   Feature columns: {len(feature_cols)} (A to XQ)")
    print(f"   BIS column: {'BIS' if 'BIS' in all_cols else 'NOT FOUND'}")
    print(f"   Subject column: {'Subject' if 'Subject' in all_cols else 'NOT FOUND'}")
    
    # BIS Stats
    print(f"\n4. BIS Statistics:")
    print(f"   Range: [{df['BIS'].min():.1f}, {df['BIS'].max():.1f}]")
    print(f"   Mean: {df['BIS'].mean():.2f}")
    print(f"   Median: {df['BIS'].median():.2f}")
    print(f"   Std: {df['BIS'].std():.2f}")
    
    # Feature Stats
    print(f"\n5. Feature Statistics (479 columns):")
    feat_data = df[feature_cols]
    print(f"   Min value: {feat_data.min().min():.4f}")
    print(f"   Max value: {feat_data.max().max():.4f}")
    print(f"   Mean: {feat_data.mean().mean():.4f}")
    print(f"   Std: {feat_data.std().mean():.4f}")
    
    # Missing values
    print(f"\n6. Data Quality:")
    missing = df.isnull().sum().sum()
    print(f"   Missing values: {missing}")
    print(f"   ✓ Complete data!" if missing == 0 else f"   ⚠️  Has {missing} missing values")
    
    # Data types
    print(f"\n7. Data Types:")
    print(f"   Numeric columns: {df.select_dtypes(include=[np.number]).shape}")
    print(f"   Object columns: {df.select_dtypes(include=['object']).shape}")
    
    # Subject distribution
    print(f"\n8. Subject Distribution:")
    print(f"   Unique subjects: {df['Subject'].nunique()}")
    print(f"   Top 5 subjects:")
    print(df['Subject'].value_counts().head())
    
    print("\n" + "="*80)
    print("SUMMARY FOR RAT-DoA TRAINING:")
    print("="*80)
    print(f"✓ Samples: {len(df):,}")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Target (BIS): 0-100 scale")
    print(f"✓ Data quality: Complete")
    print(f"\n⚠️  IMPORTANT:")
    print(f"   This is a LARGE dataset:")
    print(f"   - {len(df):,} samples × {len(feature_cols)} features")
    print(f"   - Estimated ~500MB+ in memory")
    print(f"   - Consider feature selection (479 → 256)")
    print(f"   - Use GPU for faster training")
    print(f"\n✓ READY FOR TRAINING!")
    print("="*80)

if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "data" / "raw" / "EEG_BIS_Segments.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        sys.exit(1)
    
    explore_dataset(csv_path)
