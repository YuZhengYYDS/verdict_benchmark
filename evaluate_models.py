import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from importlib import import_module
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data.dataset import get_dataloaders, load_mat_data
from utils.scaler import MinMaxScaler

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans'
})

class ModelEvaluator:
    """Comprehensive model evaluation and benchmarking class."""
    
    def __init__(self, data_config_path: str, results_dir: str = 'evaluation_results'):
        """
        Initialize evaluator with data configuration.
        
        Args:
            data_config_path: Path to any config file containing data settings
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data configuration
        with open(data_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.data_config = config['data']
        self.batch_size = config.get('batch_size', 32)
        self.train_ratio = config.get('train_ratio', 0.8)
        self.seed = config.get('seed', 42)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model registry
        self.models = {}
        self.test_predictions = {}
        self.test_targets = None
        
    def discover_models(self) -> Dict[str, Dict]:
        """
        Discover all available trained models from checkpoints and configs.
        
        Returns:
            Dictionary mapping model names to their configurations
        """
        model_configs = {}
        configs_dir = 'configs'
        checkpoints_dir = 'checkpoints'
        
        # Get all available checkpoints
        available_checkpoints = {}
        for checkpoint_file in os.listdir(checkpoints_dir):
            if checkpoint_file.endswith('_best.pt'):
                model_name = checkpoint_file.replace('_best.pt', '')
                available_checkpoints[model_name] = os.path.join(checkpoints_dir, checkpoint_file)
        
        # Try to match config files with checkpoints
        for config_file in os.listdir(configs_dir):
            if config_file.endswith('.yaml'):
                config_name = config_file.replace('.yaml', '')
                config_path = os.path.join(configs_dir, config_file)
                
                # Try exact match first
                if config_name in available_checkpoints:
                    checkpoint_path = available_checkpoints[config_name]
                    model_name = config_name
                else:
                    # Try to find a checkpoint that starts with the config model type
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    model_type = config.get('model', {}).get('type', '')
                    checkpoint_path = None
                    model_name = None
                    
                    # Look for checkpoints that match the model type
                    for checkpoint_name in available_checkpoints.keys():
                        if checkpoint_name == model_type or checkpoint_name.startswith(f"{model_type}_"):
                            checkpoint_path = available_checkpoints[checkpoint_name]
                            model_name = checkpoint_name
                            break
                
                if checkpoint_path and model_name:
                    config_path = os.path.join(configs_dir, config_file)
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    model_configs[model_name] = {
                        'config': config,
                        'checkpoint_path': checkpoint_path,
                        'config_path': config_path
                    }
                    
        print(f"Discovered {len(model_configs)} trained models: {list(model_configs.keys())}")
        return model_configs
    
    def load_model(self, model_name: str, config: Dict) -> torch.nn.Module:
        """Load a trained model from checkpoint."""
        try:
            # Import model class
            module = import_module(f"models.{config['model']['type']}")
            Model = getattr(module, config['model']['class_name'])
            
            # Get input/output dimensions from data
            X, y = load_mat_data(self.data_config['mat_path'])
            input_dim = X.shape[1]
            output_dim = y.shape[1]
            
            # Instantiate model
            model = Model(
                input_dim=input_dim,
                output_dim=output_dim,
                **config['model']['params']
            )
            
            # Load weights
            checkpoint_path = os.path.join('checkpoints', f"{model_name}_best.pt")
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return None
    
    def prepare_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare test dataset for evaluation."""
        # Load raw data
        X, y = load_mat_data(self.data_config['mat_path'])
        total = X.shape[0]
        
        # Use same split as training
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(total, generator=generator)
        train_len = int(self.train_ratio * total)
        test_idx = perm[train_len:].numpy()
        
        X_test, y_test = X[test_idx], y[test_idx]
        
        return torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    
    def evaluate_single_model(self, model_name: str, model: torch.nn.Module, 
                            X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
        """Evaluate a single model and return metrics."""
        # Load scaler
        scaler_path = os.path.join('checkpoints', f"{model_name}_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = MinMaxScaler.load(scaler_path)
            y_test_scaled = torch.from_numpy(scaler.transform(y_test.numpy())).float()
        else:
            y_test_scaled = y_test
            scaler = None
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(0, X_test.shape[0], self.batch_size):
                batch_X = X_test[i:i+self.batch_size].to(self.device)
                model_output = model(batch_X)
                
                # Handle models that return tuples (e.g., VAE, MoE, Ensemble)
                if isinstance(model_output, tuple):
                    pred = model_output[0]
                else:
                    pred = model_output
                    
                predictions.append(pred.cpu())
        
        y_pred_scaled = torch.cat(predictions, dim=0)
        
        # Inverse transform if scaler exists
        if scaler is not None:
            y_pred = torch.from_numpy(scaler.inverse_transform(y_pred_scaled.numpy())).float()
            y_true = y_test
        else:
            y_pred = y_pred_scaled
            y_true = y_test_scaled
        
        # Store predictions for later analysis
        self.test_predictions[model_name] = y_pred.numpy()
        
        # Calculate metrics
        y_pred_np = y_pred.numpy()
        y_true_np = y_true.numpy()
        
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = mean_squared_error(y_true_np, y_pred_np)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true_np, y_pred_np)
        metrics['r2'] = r2_score(y_true_np, y_pred_np)
        
        # Per-parameter metrics
        n_params = y_true_np.shape[1]
        metrics['per_param_r2'] = []
        metrics['per_param_mse'] = []
        metrics['per_param_mae'] = []
        
        for i in range(n_params):
            metrics['per_param_r2'].append(r2_score(y_true_np[:, i], y_pred_np[:, i]))
            metrics['per_param_mse'].append(mean_squared_error(y_true_np[:, i], y_pred_np[:, i]))
            metrics['per_param_mae'].append(mean_absolute_error(y_true_np[:, i], y_pred_np[:, i]))
        
        # Statistical tests
        try:
            # Pearson correlation
            correlations = []
            p_values = []
            for i in range(n_params):
                corr, p_val = stats.pearsonr(y_true_np[:, i], y_pred_np[:, i])
                correlations.append(corr)
                p_values.append(p_val)
            metrics['correlations'] = correlations
            metrics['correlation_p_values'] = p_values
        except:
            metrics['correlations'] = [0] * n_params
            metrics['correlation_p_values'] = [1] * n_params
        
        return metrics
    
    def run_evaluation(self):
        """Run comprehensive evaluation of all discovered models."""
        print("Starting model evaluation...")
        
        # Discover available models
        model_configs = self.discover_models()
        
        if not model_configs:
            print("No trained models found!")
            return
        
        # Prepare test data
        X_test, y_test = self.prepare_test_data()
        self.test_targets = y_test.numpy()
        print(f"Test set size: {X_test.shape[0]} samples")
        
        # Evaluate each model
        results = {}
        for model_name, model_info in model_configs.items():
            print(f"Evaluating {model_name}...")
            
            try:
                model = self.load_model(model_name, model_info['config'])
                if model is None:
                    print(f"  ❌ Failed to load {model_name}")
                    continue
                    
                metrics = self.evaluate_single_model(model_name, model, X_test, y_test)
                results[model_name] = metrics
                
                print(f"  ✅ R² Score: {metrics['r2']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
            except Exception as e:
                print(f"  ❌ Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.results = results
        return results
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison plots with best model highlighting."""
        if not hasattr(self, 'results'):
            print("No evaluation results found. Run evaluation first.")
            return
        
        # Extract metrics for plotting
        models = list(self.results.keys())
        metrics_data = {
            'Model': [],
            'R²': [],
            'RMSE': [],
            'MAE': [],
            'MSE': []
        }
        
        for model in models:
            metrics_data['Model'].append(model)
            metrics_data['R²'].append(self.results[model]['r2'])
            metrics_data['RMSE'].append(self.results[model]['rmse'])
            metrics_data['MAE'].append(self.results[model]['mae'])
            metrics_data['MSE'].append(self.results[model]['mse'])
        
        df = pd.DataFrame(metrics_data)
        
        # Find best performing models for each metric
        best_r2_idx = df['R²'].idxmax()
        best_rmse_idx = df['RMSE'].idxmin()
        best_mae_idx = df['MAE'].idxmin()
        best_mse_idx = df['MSE'].idxmin()
        
        # Create color palettes with highlighting
        def create_highlight_colors(best_idx, n_models):
            colors = ['lightblue'] * n_models
            colors[best_idx] = 'gold'  # Highlight best model in gold
            return colors
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison - Best Models Highlighted', fontsize=16, fontweight='bold')
        
        # R² Score
        r2_colors = create_highlight_colors(best_r2_idx, len(models))
        bars1 = axes[0,0].bar(range(len(models)), df['R²'], color=r2_colors, edgecolor='black', linewidth=1)
        axes[0,0].set_title(f'R² Score (Higher is Better)\nBest: {df.loc[best_r2_idx, "Model"]} ({df.loc[best_r2_idx, "R²"]:.4f})')
        axes[0,0].set_xticks(range(len(models)))
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE
        rmse_colors = create_highlight_colors(best_rmse_idx, len(models))
        bars2 = axes[0,1].bar(range(len(models)), df['RMSE'], color=rmse_colors, edgecolor='black', linewidth=1)
        axes[0,1].set_title(f'Root Mean Square Error (Lower is Better)\nBest: {df.loc[best_rmse_idx, "Model"]} ({df.loc[best_rmse_idx, "RMSE"]:.4f})')
        axes[0,1].set_xticks(range(len(models)))
        axes[0,1].set_xticklabels(models, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # MAE
        mae_colors = create_highlight_colors(best_mae_idx, len(models))
        bars3 = axes[1,0].bar(range(len(models)), df['MAE'], color=mae_colors, edgecolor='black', linewidth=1)
        axes[1,0].set_title(f'Mean Absolute Error (Lower is Better)\nBest: {df.loc[best_mae_idx, "Model"]} ({df.loc[best_mae_idx, "MAE"]:.4f})')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # MSE
        mse_colors = create_highlight_colors(best_mse_idx, len(models))
        bars4 = axes[1,1].bar(range(len(models)), df['MSE'], color=mse_colors, edgecolor='black', linewidth=1)
        axes[1,1].set_title(f'Mean Square Error (Lower is Better)\nBest: {df.loc[best_mse_idx, "Model"]} ({df.loc[best_mse_idx, "MSE"]:.4f})')
        axes[1,1].set_xticks(range(len(models)))
        axes[1,1].set_xticklabels(models, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                          f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', edgecolor='black', label='Best Model'),
            Patch(facecolor='lightblue', edgecolor='black', label='Other Models')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print best models summary
        print("\n" + "="*60)
        print("BEST PERFORMING MODELS BY METRIC")
        print("="*60)
        print(f"Best R² Score:  {df.loc[best_r2_idx, 'Model']:<15} ({df.loc[best_r2_idx, 'R²']:.4f})")
        print(f"Best RMSE:      {df.loc[best_rmse_idx, 'Model']:<15} ({df.loc[best_rmse_idx, 'RMSE']:.4f})")
        print(f"Best MAE:       {df.loc[best_mae_idx, 'Model']:<15} ({df.loc[best_mae_idx, 'MAE']:.4f})")
        print(f"Best MSE:       {df.loc[best_mse_idx, 'Model']:<15} ({df.loc[best_mse_idx, 'MSE']:.4f})")
        print("="*60)
        
        return df
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for per-parameter performance."""
        if not hasattr(self, 'results'):
            return
        
        models = list(self.results.keys())
        n_params = len(self.results[models[0]]['per_param_r2'])
        
        # Get parameter names from data config
        param_names = self.data_config.get('parameter_names', [f'P{i+1}' for i in range(n_params)])
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(models), n_params))
        for i, model in enumerate(models):
            corr_matrix[i, :] = self.results[model]['correlations']
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=param_names,
                   yticklabels=models,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   fmt='.3f',
                   cbar_kws={'label': 'Pearson Correlation'})
        
        plt.title('Per-Parameter Correlation Matrix\n(Model vs Ground Truth)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Parameters')
        plt.ylabel('Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_prediction_scatter_plots(self, max_models_per_plot: int = 6):
        """Create scatter plots of predictions vs ground truth."""
        if not hasattr(self, 'results'):
            return
        
        models = list(self.test_predictions.keys())
        n_params = self.test_targets.shape[1]
        
        # Get parameter names from data config
        param_names = self.data_config.get('parameter_names', [f'P{i+1}' for i in range(n_params)])
        
        # Create plots in batches if too many models
        model_batches = [models[i:i+max_models_per_plot] 
                        for i in range(0, len(models), max_models_per_plot)]
        
        for batch_idx, model_batch in enumerate(model_batches):
            fig, axes = plt.subplots(len(model_batch), n_params, 
                                   figsize=(4*n_params, 3*len(model_batch)))
            
            if len(model_batch) == 1:
                axes = axes.reshape(1, -1)
            if n_params == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Predictions vs Ground Truth - Batch {batch_idx + 1}', 
                        fontsize=16, fontweight='bold')
            
            for i, model_name in enumerate(model_batch):
                predictions = self.test_predictions[model_name]
                
                for j in range(n_params):
                    ax = axes[i, j]
                    
                    # Scatter plot
                    ax.scatter(self.test_targets[:, j], predictions[:, j], 
                             alpha=0.6, s=20)
                    
                    # Perfect prediction line
                    min_val = min(self.test_targets[:, j].min(), predictions[:, j].min())
                    max_val = max(self.test_targets[:, j].max(), predictions[:, j].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    # Labels and title
                    ax.set_xlabel('Ground Truth')
                    ax.set_ylabel('Predictions')
                    ax.set_title(f'{model_name} - {param_names[j]}\n'
                               f'R² = {self.results[model_name]["per_param_r2"][j]:.3f}')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'scatter_plots_batch_{batch_idx+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_error_distribution_plots(self):
        """Create error distribution plots for each model."""
        if not hasattr(self, 'results'):
            return
        
        models = list(self.test_predictions.keys())
        n_models = len(models)
        
        # Calculate residuals for each model
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        elif (n_models + 1) // 2 == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
        
        for idx, model_name in enumerate(models):
            row = idx // ((n_models + 1) // 2)
            col = idx % ((n_models + 1) // 2)
            
            if axes.ndim == 1:
                ax = axes[idx]
            else:
                ax = axes[row, col]
            
            predictions = self.test_predictions[model_name]
            residuals = (predictions - self.test_targets).flatten()
            
            # Histogram of residuals
            ax.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.8)
            ax.set_xlabel('Residuals (Predicted - True)')
            ax.set_ylabel('Density')
            ax.set_title(f'{model_name}\nMean: {np.mean(residuals):.4f}, '
                        f'Std: {np.std(residuals):.4f}')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes.flat)):
            axes.flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_ranking_table(self):
        """Create a comprehensive ranking table of all models."""
        if not hasattr(self, 'results'):
            return
        
        # Prepare data for ranking
        ranking_data = []
        for model_name, metrics in self.results.items():
            ranking_data.append({
                'Model': model_name,
                'R² Score': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MSE': metrics['mse'],
                'Mean Correlation': np.mean(metrics['correlations'])
            })
        
        df = pd.DataFrame(ranking_data)
        
        # Create rankings (1 = best)
        df['R² Rank'] = df['R² Score'].rank(ascending=False)
        df['RMSE Rank'] = df['RMSE'].rank(ascending=True)
        df['MAE Rank'] = df['MAE'].rank(ascending=True)
        df['MSE Rank'] = df['MSE'].rank(ascending=True)
        df['Correlation Rank'] = df['Mean Correlation'].rank(ascending=False)
        
        # Overall rank (average of all ranks)
        rank_cols = ['R² Rank', 'RMSE Rank', 'MAE Rank', 'MSE Rank', 'Correlation Rank']
        df['Overall Rank'] = df[rank_cols].mean(axis=1)
        df['Overall Position'] = df['Overall Rank'].rank()
        
        # Sort by overall rank
        df_sorted = df.sort_values('Overall Rank').round(4)
        
        # Save to CSV
        df_sorted.to_csv(os.path.join(self.results_dir, 'model_rankings.csv'), index=False)
        
        # Display top performers
        print("\n" + "="*80)
        print("MODEL PERFORMANCE RANKING")
        print("="*80)
        print(f"{'Rank':<4} {'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Overall':<8}")
        print("-"*80)
        
        for idx, row in df_sorted.iterrows():
            print(f"{int(row['Overall Position']):<4} {row['Model']:<20} "
                  f"{row['R² Score']:<8.4f} {row['RMSE']:<8.4f} "
                  f"{row['MAE']:<8.4f} {row['Overall Rank']:<8.2f}")
        
        return df_sorted
    
    def generate_full_report(self):
        """Generate a complete evaluation report with all visualizations."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        # Run evaluation
        self.run_evaluation()
        
        # Generate all visualizations
        print("\n1. Creating performance comparison plots...")
        performance_df = self.create_performance_comparison()
        
        print("2. Creating correlation heatmap...")
        self.create_correlation_heatmap()
        
        print("3. Creating error distribution plots...")
        self.create_error_distribution_plots()
        
        print("4. Creating ranking table...")
        ranking_df = self.create_ranking_table()
        
        # Summary statistics
        print(f"\n5. Saving summary report...")
        with open(os.path.join(self.results_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write("VERDICT BENCHMARK - MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now()}\n")
            f.write(f"Number of Models Evaluated: {len(self.results)}\n")
            f.write(f"Test Set Size: {self.test_targets.shape[0]} samples\n")
            f.write(f"Number of Parameters: {self.test_targets.shape[1]}\n\n")
            
            f.write("TOP 3 PERFORMING MODELS:\n")
            f.write("-" * 30 + "\n")
            for i, (_, row) in enumerate(ranking_df.head(3).iterrows()):
                f.write(f"{i+1}. {row['Model']} (R² = {row['R² Score']:.4f})\n")
            
            f.write(f"\nBest R² Score: {ranking_df['R² Score'].max():.4f}")
            f.write(f" ({ranking_df.loc[ranking_df['R² Score'].idxmax(), 'Model']})\n")
            f.write(f"Lowest RMSE: {ranking_df['RMSE'].min():.4f}")
            f.write(f" ({ranking_df.loc[ranking_df['RMSE'].idxmin(), 'Model']})\n")
        
        print(f"\nEvaluation complete! Results saved in '{self.results_dir}' directory.")
        print("\nGenerated files:")
        print("- performance_comparison.png")
        print("- correlation_heatmap.png") 
        print("- error_distributions.png")
        print("- model_rankings.csv")
        print("- evaluation_summary.txt")


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained VERDICT models')
    parser.add_argument('--config', type=str, default='configs/mlp.yaml',
                       help='Path to config file (for data settings)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config, args.output_dir)
    
    # Generate full report
    evaluator.generate_full_report()


if __name__ == '__main__':
    main()
