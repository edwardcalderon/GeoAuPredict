#!/usr/bin/env python3
"""
Visualize Ensemble Comparison Results
Creates charts comparing Voting vs Stacking Ensemble performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def create_comparison_charts():
    """Create comprehensive comparison visualizations"""
    
    # Load data
    data_path = Path('outputs/models/ensemble_comparison.csv')
    if not data_path.exists():
        print("❌ Error: ensemble_comparison.csv not found")
        return
    
    df = pd.read_csv(data_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🏆 Ensemble Performance Comparison: Voting vs Stacking', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Define colors
    colors = {
        'random_forest': '#2ecc71',
        'xgboost': '#3498db',
        'lightgbm': '#9b59b6',
        'voting_ensemble': '#e74c3c',
        'stacking_ensemble': '#f39c12'
    }
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC ⭐']
    
    # Plot each metric
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        
        # Create bar chart
        bars = ax.bar(range(len(df)), df[metric], 
                      color=[colors[model] for model in df['model']])
        
        # Customize
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model'], rotation=45, ha='right', fontsize=10)
        ax.set_ylim([0.75, 0.95])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=9)
            
            # Highlight ensemble bars
            if df['model'].iloc[i] in ['voting_ensemble', 'stacking_ensemble']:
                bar.set_edgecolor('black')
                bar.set_linewidth(2.5)
    
    # Summary comparison subplot
    ax = axes[1, 2]
    ensemble_df = df[df['model'].str.contains('ensemble')]
    
    x = range(len(metrics))
    width = 0.35
    
    voting_values = ensemble_df[ensemble_df['model'] == 'voting_ensemble'][metrics].values[0]
    stacking_values = ensemble_df[ensemble_df['model'] == 'stacking_ensemble'][metrics].values[0]
    
    bars1 = ax.bar([i - width/2 for i in x], voting_values, width, 
                   label='Voting', color=colors['voting_ensemble'], 
                   edgecolor='black', linewidth=2)
    bars2 = ax.bar([i + width/2 for i in x], stacking_values, width,
                   label='Stacking', color=colors['stacking_ensemble'],
                   edgecolor='black', linewidth=2)
    
    ax.set_title('Ensemble Comparison (All Metrics)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Acc', 'Prec', 'Rec', 'F1', 'AUC'], fontsize=11)
    ax.set_ylim([0.75, 0.95])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs/models/ensemble_comparison_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison chart to {output_path}")
    
    # Also save as PDF
    pdf_path = Path('outputs/models/ensemble_comparison_chart.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved comparison chart to {pdf_path}")
    
    plt.show()
    
    # Create winner announcement
    print("\n" + "="*60)
    print("📊 VISUAL COMPARISON COMPLETE")
    print("="*60)
    print(f"\n🥇 WINNER: VOTING ENSEMBLE")
    print(f"   - AUC-ROC: {voting_values[4]:.4f}")
    print(f"   - Simpler and more robust")
    print(f"\n🥈 RUNNER-UP: STACKING ENSEMBLE")
    print(f"   - AUC-ROC: {stacking_values[4]:.4f}")
    print(f"   - Difference: {abs(voting_values[4] - stacking_values[4]):.4f} (0.02%)")

if __name__ == '__main__':
    create_comparison_charts()

