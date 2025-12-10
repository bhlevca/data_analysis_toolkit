#!/usr/bin/env python
"""
Example: Basic Data Analysis
============================

This example demonstrates how to use the toolkit programmatically
without the GUI interface.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')

from data_toolkit import (
    DataLoader,
    StatisticalAnalysis,
    MLModels,
    VisualizationMethods
)


def main():
    # Generate sample data
    print("=" * 60)
    print("ADVANCED DATA ANALYSIS TOOLKIT - Example")
    print("=" * 60)
    
    np.random.seed(42)
    n = 200
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 2.5 * x1 - 1.5 * x2 + np.random.randn(n) * 0.5
    
    df = pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'target': y
    })
    
    print(f"\nGenerated dataset with {len(df)} samples")
    print(df.head())
    
    # Initialize modules
    stats = StatisticalAnalysis(df)
    ml = MLModels(df)
    viz = VisualizationMethods(df)
    
    features = ['feature_1', 'feature_2']
    target = 'target'
    
    # 1. Descriptive Statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    desc = stats.descriptive_stats(features + [target])
    print(desc)
    
    # 2. Correlation Analysis
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    corr = stats.correlation_matrix(features + [target])
    print(corr)
    
    # 3. Train ML Model
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)
    results = ml.train_model(features, target, 'Linear Regression')
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print("\nCoefficients:")
    for feat, coef in results['coefficients'].items():
        print(f"  {feat}: {coef:.4f}")
    
    # 4. Feature Importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("=" * 60)
    importance = ml.feature_importance(features, target)
    for feat, imp in importance.items():
        print(f"  {feat}: {imp:.4f}")
    
    # 5. Visualizations
    print("\n" + "=" * 60)
    print("GENERATING PLOTS...")
    print("=" * 60)
    
    # Scatter plot
    fig1 = viz.scatter_plot('feature_1', target)
    plt.savefig('scatter_plot.png', dpi=150, bbox_inches='tight')
    print("  Saved: scatter_plot.png")
    
    # Correlation heatmap
    fig2 = viz.correlation_heatmap(features + [target])
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("  Saved: correlation_heatmap.png")
    
    # Predictions vs Actual
    fig3 = ml.plot_predictions_vs_actual()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("  Saved: predictions.png")
    
    plt.close('all')
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE!")
    print("=" * 60)
    print("\nTo run the GUI application:")
    print("  python -m data_toolkit.main_gui")
    print("\nOr install and use the command:")
    print("  pip install -e .")
    print("  data-toolkit")


if __name__ == '__main__':
    main()
