import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from evaluate_models import ModelEvaluator
from utils.metrics import statistical_significance_test, bootstrap_confidence_interval

# Enhanced plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 100
})

class AdvancedModelEvaluator(ModelEvaluator):
    """Extended evaluator with advanced statistical analysis and publication-ready plots."""
    
    def __init__(self, data_config_path: str, results_dir: str = 'advanced_evaluation'):
        super().__init__(data_config_path, results_dir)
        self.significance_tests = {}
        
    def perform_pairwise_significance_tests(self):
        """Perform pairwise statistical significance tests between all models."""
        models = list(self.test_predictions.keys())
        n_models = len(models)
        
        # Initialize results matrices
        p_values_matrix = np.ones((n_models, n_models))
        effect_sizes_matrix = np.zeros((n_models, n_models))
        
        print("\nPerforming pairwise significance tests...")
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                model1, model2 = models[i], models[j]
                
                test_results = statistical_significance_test(
                    self.test_predictions[model1],
                    self.test_predictions[model2],
                    self.test_targets
                )
                
                p_values_matrix[i, j] = test_results['paired_t_test']['p_value']
                p_values_matrix[j, i] = test_results['paired_t_test']['p_value']
                
                effect_sizes_matrix[i, j] = test_results['effect_size']['cohens_d']
                effect_sizes_matrix[j, i] = -test_results['effect_size']['cohens_d']
                
                self.significance_tests[f"{model1}_vs_{model2}"] = test_results
        
        # Create significance test visualization
        self._plot_significance_matrix(models, p_values_matrix, effect_sizes_matrix)
        
        return p_values_matrix, effect_sizes_matrix
    
    def _plot_significance_matrix(self, models: List[str], p_values: np.ndarray, 
                                effect_sizes: np.ndarray):
        """Plot significance test results as heatmaps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # P-values heatmap
        mask = np.triu(np.ones_like(p_values))
        sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   xticklabels=models, yticklabels=models,
                   mask=mask, cbar_kws={'label': 'p-value'}, ax=ax1)
        ax1.set_title('Statistical Significance (p-values)\nLower is more significant')
        
        # Effect sizes heatmap
        mask = np.triu(np.ones_like(effect_sizes))
        sns.heatmap(effect_sizes, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   xticklabels=models, yticklabels=models,
                   mask=mask, cbar_kws={'label': "Cohen's d"}, ax=ax2)
        ax2.set_title('Effect Sizes (Cohen\'s d)\nPositive favors row model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'significance_tests.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_confidence_intervals_plot(self):
        """Create confidence intervals plot for model performance with best model highlighting."""
        if not hasattr(self, 'results'):
            return
        
        models = list(self.results.keys())
        metrics = ['R²', 'RMSE', 'MAE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for metric_idx, metric in enumerate(metrics):
            ci_data = []
            
            for model in models:
                pred = self.test_predictions[model]
                target = self.test_targets
                
                if metric == 'R²':
                    metric_func = lambda p, t: stats.pearsonr(p.flatten(), t.flatten())[0] ** 2
                elif metric == 'RMSE':
                    metric_func = lambda p, t: np.sqrt(np.mean((p - t) ** 2))
                else:  # MAE
                    metric_func = lambda p, t: np.mean(np.abs(p - t))
                
                value, lower, upper = bootstrap_confidence_interval(
                    pred, target, metric_func, n_bootstrap=1000
                )
                
                ci_data.append({
                    'Model': model,
                    'Value': value,
                    'Lower': lower,
                    'Upper': upper,
                    'Error_Lower': value - lower,
                    'Error_Upper': upper - value
                })
            
            df_ci = pd.DataFrame(ci_data)
            
            # Find best model for this metric
            if metric == 'R²':
                best_idx = df_ci['Value'].idxmax()
            else:  # For RMSE and MAE, lower is better
                best_idx = df_ci['Value'].idxmin()
            
            # Create colors with highlighting
            colors = ['lightblue'] * len(models)
            colors[best_idx] = 'gold'
            
            # Plot with error bars
            x_pos = range(len(models))
            bars = axes[metric_idx].bar(x_pos, df_ci['Value'], 
                                      yerr=[df_ci['Error_Lower'], df_ci['Error_Upper']],
                                      capsize=5, alpha=0.8, edgecolor='black',
                                      color=colors, linewidth=1)
            
            axes[metric_idx].set_xlabel('Models')
            axes[metric_idx].set_ylabel(metric)
            best_model = df_ci.loc[best_idx, 'Model']
            best_value = df_ci.loc[best_idx, 'Value']
            axes[metric_idx].set_title(f'{metric} with 95% Confidence Intervals\nBest: {best_model} ({best_value:.4f})')
            axes[metric_idx].set_xticks(x_pos)
            axes[metric_idx].set_xticklabels(models, rotation=45)
            axes[metric_idx].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[metric_idx].text(bar.get_x() + bar.get_width()/2., height + df_ci.loc[i, 'Error_Upper'] + height*0.01,
                                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', edgecolor='black', label='Best Model'),
            Patch(facecolor='lightblue', edgecolor='black', label='Other Models')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(os.path.join(self.results_dir, 'confidence_intervals.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_parameter_wise_comparison(self):
        """Create detailed parameter-wise comparison plots with improved readability for many models."""
        if not hasattr(self, 'results'):
            return
        
        models = list(self.results.keys())
        n_params = len(self.results[models[0]]['per_param_r2'])
        
        # Get actual parameter names from data configuration
        param_names = self.data_config.get('parameter_names', [f'P{i+1}' for i in range(n_params)])
        
        # Prepare data
        param_data = []
        for model in models:
            for param_idx in range(n_params):
                param_data.append({
                    'Model': model,
                    'Parameter': param_names[param_idx],  # Use actual parameter names
                    'R²': self.results[model]['per_param_r2'][param_idx],
                    'MSE': self.results[model]['per_param_mse'][param_idx],
                    'MAE': self.results[model]['per_param_mae'][param_idx],
                    'Correlation': self.results[model]['correlations'][param_idx]
                })
        
        df_params = pd.DataFrame(param_data)
        
        # Create two separate figures for better readability
        # Figure 1: R² and Correlation (most important metrics)
        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 8))
        fig1.suptitle('Parameter-wise Model Performance: Primary Metrics', fontsize=16, fontweight='bold')
        
        # R² with heatmap style for better visualization
        df_pivot_r2 = df_params.pivot(index='Model', columns='Parameter', values='R²')
        sns.heatmap(df_pivot_r2, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'R² Score'}, ax=axes1[0])
        axes1[0].set_title('R² Score by Model and Parameter')
        axes1[0].set_xlabel('Parameter')
        axes1[0].set_ylabel('Model')
        
        # Correlation heatmap
        df_pivot_corr = df_params.pivot(index='Model', columns='Parameter', values='Correlation')
        sns.heatmap(df_pivot_corr, annot=True, fmt='.3f', cmap='plasma',
                   cbar_kws={'label': 'Correlation'}, ax=axes1[1])
        axes1[1].set_title('Correlation by Model and Parameter')
        axes1[1].set_xlabel('Parameter')
        axes1[1].set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_wise_primary_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Error metrics (MSE and MAE)
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
        fig2.suptitle('Parameter-wise Model Performance: Error Metrics', fontsize=16, fontweight='bold')
        
        # MSE heatmap (log scale for better visualization)
        df_pivot_mse = df_params.pivot(index='Model', columns='Parameter', values='MSE')
        # Use log scale for MSE due to potentially large values
        sns.heatmap(np.log10(df_pivot_mse + 1e-10), annot=df_pivot_mse.values, fmt='.2e', 
                   cmap='Reds', cbar_kws={'label': 'log₁₀(MSE)'}, ax=axes2[0])
        axes2[0].set_title('MSE by Model and Parameter (log scale)')
        axes2[0].set_xlabel('Parameter')
        axes2[0].set_ylabel('Model')
        
        # MAE heatmap
        df_pivot_mae = df_params.pivot(index='Model', columns='Parameter', values='MAE')
        sns.heatmap(df_pivot_mae, annot=True, fmt='.3f', cmap='Oranges',
                   cbar_kws={'label': 'MAE'}, ax=axes2[1])
        axes2[1].set_title('MAE by Model and Parameter')
        axes2[1].set_xlabel('Parameter')
        axes2[1].set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_wise_error_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 3: Summary statistics for each parameter
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Parameter Performance Summary Statistics', fontsize=16, fontweight='bold')
        
        # Best performing model per parameter for each metric
        metrics = ['R²', 'MSE', 'MAE', 'Correlation']
        metric_titles = ['Best R² by Parameter', 'Best MSE by Parameter', 
                        'Best MAE by Parameter', 'Best Correlation by Parameter']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes3[idx//2, idx%2]
            
            # Find best model for each parameter
            if metric in ['MSE', 'MAE']:
                best_per_param = df_params.groupby('Parameter')[metric].idxmin()
            else:
                best_per_param = df_params.groupby('Parameter')[metric].idxmax()
            
            best_models = df_params.loc[best_per_param, ['Parameter', 'Model', metric]]
            
            # Create bar plot
            bars = ax.bar(best_models['Parameter'], best_models[metric], 
                         color=sns.color_palette("husl", len(best_models)))
            ax.set_title(title)
            ax.set_xlabel('Parameter')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add model labels on bars
            for i, (bar, model) in enumerate(zip(bars, best_models['Model'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       model, ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_wise_best_performers.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also save the legacy comparison plot but with improved styling
        self._create_legacy_parameter_comparison(df_params)
        
        return df_params
    
    def _create_legacy_parameter_comparison(self, df_params):
        """Create the original style comparison but with improved readability."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Parameter-wise Model Performance (Detailed View)', fontsize=16, fontweight='bold')
        
        # Use smaller points and no overlapping stripplot
        plot_params = {'size': 4, 'alpha': 0.6}
        
        # R² by parameter - use violin plot for better density visualization
        sns.violinplot(data=df_params, x='Parameter', y='R²', ax=axes[0,0], 
                      inner='quart', palette='viridis')
        axes[0,0].set_title('R² Score Distribution by Parameter')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MSE by parameter - box plot only due to wide range
        sns.boxplot(data=df_params, x='Parameter', y='MSE', ax=axes[0,1])
        axes[0,1].set_title('MSE Distribution by Parameter')
        axes[0,1].set_yscale('log')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MAE by parameter
        sns.violinplot(data=df_params, x='Parameter', y='MAE', ax=axes[1,0],
                      inner='quart', palette='plasma')
        axes[1,0].set_title('MAE Distribution by Parameter')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Correlation by parameter
        sns.violinplot(data=df_params, x='Parameter', y='Correlation', ax=axes[1,1],
                      inner='quart', palette='coolwarm')
        axes[1,1].set_title('Correlation Distribution by Parameter')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parameter_wise_detailed_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_complexity_analysis(self):
        """Analyze model complexity vs performance trade-offs."""
        if not hasattr(self, 'results'):
            return
        
        # Get model configurations to estimate complexity
        model_configs = self.discover_models()
        complexity_data = []
        
        for model_name in self.results.keys():
            if model_name in model_configs:
                config = model_configs[model_name]['config']
                
                # Estimate parameter count (simplified)
                complexity_score = self._estimate_model_complexity(config)
                
                complexity_data.append({
                    'Model': model_name,
                    'Complexity': complexity_score,
                    'R²': self.results[model_name]['r2'],
                    'RMSE': self.results[model_name]['rmse'],
                    'MAE': self.results[model_name]['mae']
                })
        
        df_complexity = pd.DataFrame(complexity_data)
        
        # Create complexity vs performance plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['R²', 'RMSE', 'MAE']
        colors = ['green', 'red', 'orange']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            axes[i].scatter(df_complexity['Complexity'], df_complexity[metric], 
                          s=100, alpha=0.7, c=color, edgecolors='black')
            
            # Add model labels
            for _, row in df_complexity.iterrows():
                axes[i].annotate(row['Model'], 
                               (row['Complexity'], row[metric]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8)
            
            # Fit trend line
            if len(df_complexity) > 2:
                z = np.polyfit(df_complexity['Complexity'], df_complexity[metric], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_complexity['Complexity'].min(), 
                                    df_complexity['Complexity'].max(), 100)
                axes[i].plot(x_trend, p(x_trend), '--', alpha=0.8, color='gray')
            
            axes[i].set_xlabel('Model Complexity (Estimated Parameters)')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'Model Complexity vs {metric}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'complexity_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_complexity
    
    def _estimate_model_complexity(self, config: Dict) -> int:
        """Estimate model complexity based on configuration."""
        model_type = config['model']['type']
        params = config['model']['params']
        
        # Simplified complexity estimation
        if model_type == 'mlp':
            hidden_dims = params.get('hidden_dims', [])
            # Rough parameter count for MLP
            complexity = sum(hidden_dims) * 2  # Simplified
        elif model_type == 'cnn':
            complexity = 1000  # Default estimate
        elif model_type == 'rnn':
            hidden_size = params.get('hidden_size', 64)
            num_layers = params.get('num_layers', 1)
            complexity = hidden_size * num_layers * 4  # Rough LSTM estimate
        elif model_type == 'transformer':
            d_model = params.get('d_model', 128)
            nhead = params.get('nhead', 8)
            num_layers = params.get('num_layers', 2)
            complexity = d_model * nhead * num_layers * 4
        else:
            complexity = 500  # Default estimate
        
        return complexity
    
    def generate_latex_table(self):
        """Generate LaTeX table for publication."""
        if not hasattr(self, 'results'):
            return
        
        # Prepare data
        latex_data = []
        for model_name, metrics in self.results.items():
            latex_data.append({
                'Model': model_name.replace('_', r'\_'),
                'R²': f"{metrics['r2']:.4f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'Mean Corr.': f"{np.mean(metrics['correlations']):.4f}"
            })
        
        df_latex = pd.DataFrame(latex_data)
        
        # Sort by R² score
        df_latex['R²_num'] = df_latex['R²'].astype(float)
        df_latex = df_latex.sort_values('R²_num', ascending=False).drop('R²_num', axis=1)
        
        # Generate LaTeX code
        latex_table = df_latex.to_latex(index=False, escape=False, 
                                       caption='Model Performance Comparison on VERDICT Dataset',
                                       label='tab:model_performance',
                                       column_format='lcccc')
        
        # Save to file
        with open(os.path.join(self.results_dir, 'results_table.tex'), 'w') as f:
            f.write(latex_table)
        
        print("LaTeX table saved to results_table.tex")
        return latex_table
    
    def generate_comprehensive_report(self):
        """Generate the most comprehensive evaluation report."""
        print("\n" + "="*70)
        print("ADVANCED MODEL EVALUATION WITH STATISTICAL ANALYSIS")
        print("="*70)
        
        # Basic evaluation
        self.run_evaluation()
        
        # Basic visualizations
        print("\n1. Creating basic performance comparisons...")
        self.create_performance_comparison()
        self.create_correlation_heatmap()
        
        # Advanced analysis
        print("2. Performing statistical significance tests...")
        self.perform_pairwise_significance_tests()
        
        print("3. Creating confidence interval plots...")
        self.create_confidence_intervals_plot()
        
        print("4. Creating parameter-wise analysis...")
        self.create_parameter_wise_comparison()
        
        print("5. Analyzing model complexity trade-offs...")
        self.create_model_complexity_analysis()
        
        print("6. Creating publication-ready visualizations...")
        self.create_error_distribution_plots()
        
        print("7. Generating ranking table...")
        self.create_ranking_table()
        
        print("8. Generating LaTeX table...")
        self.generate_latex_table()
        
        # Save comprehensive summary
        self._save_advanced_summary()
        
        print(f"\nAdvanced evaluation complete! Results saved in '{self.results_dir}' directory.")
        print("\nGenerated files include:")
        print("- All basic evaluation plots")
        print("- significance_tests.png (statistical comparison)")
        print("- confidence_intervals.png (95% CI for metrics)")
        print("- parameter_wise_comparison.png (detailed parameter analysis)")
        print("- complexity_analysis.png (complexity vs performance)")
        print("- results_table.tex (LaTeX table for papers)")
        print("- advanced_evaluation_summary.txt (comprehensive report)")
    
    def _save_advanced_summary(self):
        """Save detailed summary with statistical analysis."""
        with open(os.path.join(self.results_dir, 'advanced_evaluation_summary.txt'), 'w') as f:
            f.write("VERDICT BENCHMARK - ADVANCED MODEL EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
            f.write(f"Number of Models: {len(self.results)}\n")
            f.write(f"Test Set Size: {self.test_targets.shape[0]} samples\n")
            f.write(f"Parameters: {self.test_targets.shape[1]}\n\n")
            
            # Best performers
            best_r2 = max(self.results.items(), key=lambda x: x[1]['r2'])
            best_rmse = min(self.results.items(), key=lambda x: x[1]['rmse'])
            best_mae = min(self.results.items(), key=lambda x: x[1]['mae'])
            
            f.write("BEST PERFORMING MODELS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best R² Score: {best_r2[0]} ({best_r2[1]['r2']:.4f})\n")
            f.write(f"Best RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})\n")
            f.write(f"Best MAE: {best_mae[0]} ({best_mae[1]['mae']:.4f})\n\n")
            
            # Statistical significance summary
            if hasattr(self, 'significance_tests'):
                f.write("STATISTICAL SIGNIFICANCE TESTS:\n")
                f.write("-" * 40 + "\n")
                significant_pairs = []
                for pair_name, test_result in self.significance_tests.items():
                    if test_result['paired_t_test']['significant']:
                        significant_pairs.append(pair_name)
                
                f.write(f"Significant differences found in {len(significant_pairs)} model pairs:\n")
                for pair in significant_pairs[:5]:  # Show first 5
                    f.write(f"  - {pair}\n")
                
                if len(significant_pairs) > 5:
                    f.write(f"  ... and {len(significant_pairs) - 5} more\n")
            
            f.write(f"\nFor complete statistical analysis, see significance_tests.png\n")
            f.write("For publication-ready results, see results_table.tex\n")


def main():
    """Main script for advanced model evaluation."""
    parser = argparse.ArgumentParser(description='Advanced evaluation of VERDICT models')
    parser.add_argument('--config', type=str, default='configs/mlp.yaml',
                       help='Config file path (for data settings)')
    parser.add_argument('--output-dir', type=str, default='advanced_evaluation',
                       help='Output directory for results')
    parser.add_argument('--basic-only', action='store_true',
                       help='Run only basic evaluation (faster)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator(args.config, args.output_dir)
    
    # Run evaluation
    if args.basic_only:
        evaluator.generate_full_report()
    else:
        evaluator.generate_comprehensive_report()


if __name__ == '__main__':
    main()
