# VERDICT Benchmark Model Evaluation

This directory contains comprehensive evaluation scripts for benchmarking trained VERDICT models.

## Features

### Basic Evaluation (`evaluate_models.py`)
- **Performance Comparison**: Bar plots comparing R², RMSE, MAE, MSE across all models with best model highlighting
- **Correlation Analysis**: Heatmap showing per-parameter correlations with ground truth
- **Error Distribution Analysis**: Histograms of prediction residuals
- **Model Ranking Table**: Comprehensive ranking based on multiple metrics

### Advanced Evaluation (`advanced_evaluate.py`)
- **Statistical Significance Testing**: Pairwise t-tests and Wilcoxon tests between models
- **Bootstrap Confidence Intervals**: 95% confidence intervals for all metrics
- **Parameter-wise Analysis**: Detailed breakdown of performance per parameter
- **Model Complexity Analysis**: Trade-off analysis between model complexity and performance
- **Publication-ready LaTeX Tables**: Formatted tables for research papers

## Quick Start

### Method 1: Automated (Windows)
```bash
# Run the batch script to install dependencies and run both evaluations
run_evaluation.bat
```

### Method 2: Manual

1. **Install dependencies**:
```bash
pip install -r requirements_eval.txt
```

2. **Run basic evaluation**:
```bash
python evaluate_models.py --config configs/mlp.yaml --output-dir evaluation_results
```

3. **Run advanced evaluation**:
```bash
python advanced_evaluate.py --config configs/mlp.yaml --output-dir advanced_evaluation
```

## Output Files

### Basic Evaluation Results (`evaluation_results/`)
- `performance_comparison.png` - Overall model performance comparison with best model highlighting
- `correlation_heatmap.png` - Per-parameter correlation matrix
- `error_distributions.png` - Error distribution histograms
- `model_rankings.csv` - Detailed ranking table in CSV format
- `evaluation_summary.txt` - Text summary of results

### Advanced Evaluation Results (`advanced_evaluation/`)
All basic evaluation files plus:
- `significance_tests.png` - Statistical significance test results
- `confidence_intervals.png` - Bootstrap confidence intervals
- `parameter_wise_comparison.png` - Detailed parameter analysis
- `complexity_analysis.png` - Model complexity vs performance
- `results_table.tex` - LaTeX table for publications
- `advanced_evaluation_summary.txt` - Comprehensive statistical summary

## Usage Examples

### Custom Configuration
```bash
# Use different config file for data settings
python evaluate_models.py --config configs/transformer.yaml

# Custom output directory
python evaluate_models.py --output-dir my_results
```

### Basic Evaluation Only
```bash
# For faster evaluation without statistical tests
python advanced_evaluate.py --basic-only
```

## Supported Models

The evaluation automatically discovers and evaluates all trained models with:
- Model checkpoint files in `checkpoints/` directory (`*_best.pt`)
- Corresponding configuration files in `configs/` directory (`*.yaml`)
- Associated scaler files (`*_scaler.pkl`)

Currently supported model types:
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- RNN/LSTM (Recurrent Neural Networks)
- Transformer
- VAE (Variational Autoencoder)
- TabNet
- SNN (Spiking Neural Network)
- Ensemble models
- MoE (Mixture of Experts)

## Generated Metrics

### Primary Metrics
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Square Error (lower is better)

### Advanced Metrics
- **Pearson Correlation**: Linear correlation with ground truth
- **Per-parameter Performance**: Individual metrics for each output parameter
- **Statistical Significance**: p-values from paired t-tests
- **Effect Sizes**: Cohen's d for practical significance
- **Bootstrap Confidence Intervals**: Uncertainty quantification

## Research Paper Integration

### LaTeX Integration
The evaluation generates a ready-to-use LaTeX table:
```latex
% Include the generated table in your paper
\input{advanced_evaluation/results_table.tex}
```

### Figure Recommendations
For research papers, we recommend including:
1. **Performance Comparison** (`performance_comparison.png`) - Shows overall model ranking with highlighting
2. **Statistical Significance** (`significance_tests.png`) - Validates differences between models
3. **Parameter-wise Analysis** (`parameter_wise_comparison.png`) - Detailed parameter breakdown
4. **Error Distributions** (`error_distributions.png`) - Model prediction quality analysis

## Troubleshooting

### Common Issues

1. **Missing models**: Ensure both `.pt` checkpoint and `.yaml` config files exist
2. **Memory errors**: Reduce batch size in the configuration file
3. **Import errors**: Install all dependencies with `pip install -r requirements_eval.txt`
4. **Path issues**: Use absolute paths in configuration files

### Model Loading Errors
If a model fails to load:
- Check that the model architecture matches the saved checkpoint
- Verify that all required parameters are in the config file
- Ensure the model class name is correctly specified

### Plotting Issues
If plots don't display:
- On Windows: Install `matplotlib` backend with `pip install matplotlib[gui]`
- For headless systems: Set `MPLBACKEND=Agg` environment variable

## Customization

### Adding Custom Metrics
Extend the `ModelEvaluator` class in `evaluate_models.py`:
```python
def custom_metric(self, pred, target):
    # Your custom metric implementation
    return metric_value
```

### Custom Visualizations
Add new plotting methods to the evaluator class:
```python
def create_custom_plot(self):
    # Your custom visualization
    plt.savefig(os.path.join(self.results_dir, 'custom_plot.png'))
```

## Configuration

The evaluation uses the data configuration from any model config file. Key settings:
- `data.mat_path`: Path to the dataset
- `batch_size`: Batch size for evaluation
- `train_ratio`: Train/test split ratio (must match training)
- `seed`: Random seed for reproducible splits

## Citation

When using these evaluation results in research papers, please cite the VERDICT benchmark and mention the specific evaluation metrics used.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure model checkpoints and configs are properly formatted
