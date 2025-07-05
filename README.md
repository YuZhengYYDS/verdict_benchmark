# VERDICT Benchmark: Deep Learning for Medical Parameter Prediction

A comprehensive benchmark suite for evaluating deep learning models on VERDICT (Vascular, Extracellular, and Restricted Diffusion for Cytometry in Tumours) parameter prediction from medical imaging data.

## 🔬 What is VERDICT?

VERDICT is an advanced diffusion MRI technique that provides quantitative biomarkers for tissue microstructure analysis. It models tissue as three distinct compartments:
- **Vascular**: Blood vessels and vasculature
- **Extracellular**: Extracellular space
- **Restricted**: Intracellular space (cells)

This benchmark evaluates how well different neural network architectures can predict these critical medical parameters from imaging features.

## 🎯 Project Overview

This benchmark provides:
- **State-of-the-art Models**: From simple MLPs to advanced architectures
- **Comprehensive Evaluation**: Statistical analysis with confidence intervals
- **Standardized Training**: Consistent protocols across all models
- **Publication-ready Results**: LaTeX tables and research-grade figures
- **Extensible Framework**: Easy to add new models and datasets

## 🏗️ Architecture Zoo

Our benchmark includes diverse neural network architectures, each with detailed documentation:

### 📊 Feedforward Networks
- **[Multi-Layer Perceptron (MLP)](docs/models/README_MLP.md)** - Simple yet effective baseline
  - 3-layer architecture with configurable activations
  - Fast training and inference
  - Excellent starting point for tabular data

- **[Simple ResNet (Residual MLP)](docs/models/README_ResidualMLP.md)** - Enhanced with skip connections
  - Solves vanishing gradient problem
  - Enables deeper network training
  - Better performance on complex patterns

### 🌊 Sequence Models
- **[Recurrent Neural Network (RNN)](docs/models/README_RNN.md)** - Temporal pattern modeling
  - LSTM/GRU variants for sequence processing
  - Adaptive input reshaping strategies
  - Captures sequential dependencies

- **[Transformer](docs/models/README_Transformer.md)** - Attention-based architecture
  - Multi-head self-attention mechanisms
  - Parallel processing capabilities
  - Global feature relationship modeling

### 🔍 Convolutional Networks
- **[Convolutional Neural Network (CNN)](docs/models/CNN_IMPROVEMENTS.md)** - Spatial pattern recognition
  - 1D convolutions for feature extraction
  - Hierarchical representation learning
  - Translation-invariant features

### 🧠 Advanced Architectures
- **[Variational Autoencoder (VAE)](docs/models/README_VAE.md)** - Probabilistic latent modeling
  - Learns compressed representations
  - Uncertainty quantification
  - Generative capabilities

- **[Mixture of Experts (MoE)](docs/models/MOE_ARCHITECTURE.md)** - Ensemble learning
  - Specialized expert networks
  - Dynamic routing mechanisms
  - Scalable model capacity

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
pip install torch torchvision torchaudio
pip install -r requirements_eval.txt

# Optional: Install in development mode
pip install -e .
```

### Installation Verification
```bash
# Check if installation is working
python -c "from models.mlp import MLP; print('✅ Models imported successfully')"
python -c "from data.dataset import VERDICTDataset; print('✅ Dataset imported successfully')"
python -c "from utils.metrics import calculate_metrics; print('✅ Utils imported successfully')"
```

### 1. Training Models
```bash
# Train individual models
python train.py --config configs/mlp.yaml
python train.py --config configs/transformer.yaml
python train.py --config configs/cnn_advanced.yaml

# Train all models (Windows)
run_evaluation.bat
```

### 2. Evaluation
```bash
# Basic evaluation
python evaluate_models.py --config configs/mlp.yaml

# Advanced statistical analysis
python advanced_evaluate.py --config configs/mlp.yaml

# Train and evaluate DenseNet model
python train.py --config configs/densenet_regressor.yaml

# Automated evaluation (recommended)
run_evaluation.bat
```

### 3. Results
Results are automatically saved to:
- `evaluation_results/` - Basic performance metrics
- `advanced_evaluation/` - Statistical analysis and publication-ready figures
- `wandb/` - Weights & Biases experiment tracking
- `checkpoints/` - Trained model weights and scalers

## � Model Implementation Status

| Model | Implementation | Config | Documentation | Status |
|-------|---------------|---------|---------------|---------|
| MLP | ✅ `mlp.py` | ✅ `mlp.yaml` | ✅ `README_MLP.md` | Ready |
| Residual MLP | ✅ `residual_mlp.py` | ✅ `residual_mlp.yaml` | ✅ `README_ResidualMLP.md` | Ready |
| RNN/LSTM | ✅ `rnn.py` | ✅ `rnn.yaml` | ✅ `README_RNN.md` | Ready |
| Transformer | ✅ `transformer.py` | ✅ `transformer.yaml` | ✅ `README_Transformer.md` | Ready |
| CNN | ✅ `cnn.py` | ✅ `cnn_advanced.yaml` | ✅ `CNN_IMPROVEMENTS.md` | Ready |
| VAE | ✅ `vae_regressor.py` | ✅ `vae_regressor.yaml` | ✅ `README_VAE.md` | Ready |
| MoE | ✅ `moe_regressor.py` | ✅ `moe_regressor.yaml` | ✅ `MOE_ARCHITECTURE.md` | Ready |

*Note: TabNet implementation is referenced in performance tables but implementation files are not yet available in the repository.*

## �📈 Performance Overview

| Model | R² Score | RMSE | Training Time | Parameters |
|-------|----------|------|---------------|------------|
| MLP | 0.527 | 0.08-0.12 | 10-20 min | ~50K |
| Residual MLP | 0.532 | 0.07-0.11 | 15-25 min | ~60K |
| RNN (LSTM) | 0.480 | 0.08-0.13 | 20-35 min | ~80K |
| Transformer | 0.524 | 0.07-0.12 | 15-30 min | ~100K |
| CNN | 0.88-0.95 | 0.06-0.10 | 25-40 min | ~120K |
| VAE | 0.463 | 0.08-0.12 | 25-40 min | ~150K |
| MoE | 0.440 | 0.05-0.09 | 45-60 min | ~200K |

*Performance ranges reflect different hyperparameter configurations and dataset splits.*

## 📊 Comprehensive Evaluation

### Basic Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Per-parameter Analysis**: Individual parameter performance

### Advanced Statistics
- **Statistical Significance**: Pairwise model comparisons
- **Bootstrap Confidence Intervals**: Uncertainty quantification
- **Effect Sizes**: Practical significance assessment

### Evaluation Documentation
- **[Evaluation Guide](docs/eval/EVALUATION_README.md)** - Complete evaluation instructions

## 🛠️ Project Structure

```
verdict_benchmark/
├── 📁 models/                   # Model implementations
│   ├── mlp.py                   # Multi-Layer Perceptron
│   ├── residual_mlp.py          # Residual MLP
│   ├── rnn.py                   # RNN/LSTM/GRU
│   ├── transformer.py           # Transformer
│   ├── cnn.py                   # Convolutional Network
│   ├── densenet_regressor.py    # DenseNet Regressor
│   ├── vae_regressor.py         # Variational Autoencoder
│   └── moe_regressor.py         # Mixture of Experts
├── 📁 configs/                  # Configuration files
│   ├── mlp.yaml                 # MLP settings
│   ├── transformer.yaml         # Transformer settings
│   ├── cnn_advanced.yaml        # CNN settings
│   ├── rnn.yaml                 # RNN settings
│   ├── residual_mlp.yaml        # Residual MLP settings
│   ├── densenet_regressor.yaml  # DenseNet settings
│   ├── vae_regressor.yaml       # VAE settings
│   └── moe_regressor.yaml       # MoE settings
├── 📁 docs/                     # Documentation
│   ├── models/                  # Model documentation
│   │   ├── README_MLP.md        # MLP guide
│   │   ├── README_ResidualMLP.md # Residual MLP guide
│   │   ├── README_RNN.md        # RNN guide
│   │   ├── README_Transformer.md # Transformer guide
│   │   ├── README_DenseNet.md   # DenseNet guide
│   │   ├── README_VAE.md        # VAE guide
│   │   ├── CNN_IMPROVEMENTS.md  # CNN enhancements
│   │   └── MOE_ARCHITECTURE.md  # MoE architecture
│   └── eval/                    # Evaluation documentation
│       └── EVALUATION_README.md # Evaluation guide
├── 📁 data/                     # Dataset utilities
│   ├── dataset.py               # Data loading
│   └── demodataset.ipynb        # Data exploration
├── 📁 utils/                    # Utility functions
│   ├── metrics.py               # Evaluation metrics
│   └── scaler.py                # Data preprocessing
├── 📁 checkpoints/              # Trained models
├── 📁 logs/                     # Training logs
├── 📄 train.py                  # Training script
├── 📄 evaluate_models.py        # Basic evaluation
├── 📄 advanced_evaluate.py      # Advanced analysis
├── 📄 run_evaluation.bat        # Automated evaluation
└── 📄 setup.py                  # Package setup
```

## 🎓 Research Applications

### Medical Imaging
- **Cancer Research**: Tumor microenvironment analysis
- **Treatment Monitoring**: Therapy response assessment
- **Diagnostic Support**: Quantitative biomarker extraction

### Machine Learning
- **Architecture Comparison**: Systematic model evaluation
- **Tabular Learning**: Benchmark for structured data
- **Medical AI**: Healthcare-specific deep learning

### Publications
This benchmark has been designed to support:
- **Reproducible Research**: Standardized evaluation protocols
- **Fair Comparison**: Consistent training and evaluation
- **Statistical Rigor**: Proper significance testing
- **Publication Quality**: LaTeX tables and figures

## 🔬 Dataset Information

### VERDICT Training Data
- **Features**: 153-dimensional imaging features
- **Targets**: 3 VERDICT parameters (vascular, extracellular, restricted)
- **Samples**: Professional medical imaging dataset
- **Preprocessing**: Standardized scaling and normalization

### Data Loading
```python
from data.dataset import VERDICTDataset
dataset = VERDICTDataset(mat_path="path/to/TrainingSet.mat")
```

## 📚 Model Documentation

Each model includes comprehensive documentation:

### Architecture Guides
- **[MLP README](docs/models/README_MLP.md)** - Simple feedforward networks
- **[Residual MLP README](docs/models/README_ResidualMLP.md)** - Skip connections and deep networks
- **[RNN README](docs/models/README_RNN.md)** - Sequence modeling with LSTM/GRU
- **[Transformer README](docs/models/README_Transformer.md)** - Attention mechanisms
- **[DenseNet README](docs/models/README_DenseNet.md)** - Dense connections and feature reuse
- **[VAE README](docs/models/README_VAE.md)** - Variational autoencoders
- **[MOE README](docs/models/MOE_ARCHITECTURE.md)** - Ensemble learning with specialized expert networks

## 🎯 Customization

### Adding New Models
1. Create model class in `models/`
2. Add configuration in `configs/`
3. Update training script imports
4. Create model-specific README

### Custom Datasets
1. Implement dataset class in `data/`
2. Update configuration files
3. Adjust input/output dimensions
4. Modify evaluation metrics if needed

### Hyperparameter Tuning
Each model includes extensive hyperparameter documentation:
- Architecture scaling guidelines
- Training parameter suggestions
- Regularization techniques
- Performance optimization tips

## 🔍 Advanced Features

### Weights & Biases Integration
```yaml
wandb_project: verdict_benchmark
wandb_run_name: model_experiment
```

### Learning Rate Scheduling
```yaml
scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 15
  T_mult: 2
  eta_min: 0.000001
```

### Early Stopping
```yaml
early_stop_patience: 40
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** your model or improvement
4. **Test** thoroughly
5. **Submit** a pull request

### Areas for Contribution
- New model architectures
- Evaluation metrics
- Visualization improvements
- Documentation enhancements
- Performance optimizations

## 📄 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{verdict_benchmark2025,
  title={VERDICT Benchmark: Deep Learning for Medical Parameter Prediction},
  author={Zheng Yu, Matteo Figini, ...},
  year={2025},
  month={July},
  url={NA},
  note={A comprehensive benchmark suite for evaluating deep learning models on VERDICT parameter prediction}
}
```

## 🔗 Related Work

- **VERDICT MRI**: Original diffusion MRI technique for tissue microstructure analysis
- **Medical AI Benchmarks**: Related benchmarks in medical imaging and deep learning
- **Tabular Learning**: Advances in neural networks for structured data
- **PyTorch Ecosystem**: Deep learning frameworks and tools

## 📞 Support

### Troubleshooting

**Common Issues:**
- **ImportError**: Make sure all dependencies are installed with `pip install -r requirements_eval.txt`
- **CUDA Issues**: Ensure PyTorch is installed with CUDA support if using GPU
- **Memory Errors**: Reduce batch size in config files for large models
- **Config Errors**: Check YAML syntax and ensure all required fields are present

**Performance Tips:**
- Use GPU for faster training (CUDA compatible)
- Adjust batch size based on available memory
- Enable mixed precision training for memory efficiency
- Use early stopping to prevent overfitting

### Documentation
- **[Evaluation Guide](docs/eval/EVALUATION_README.md)** - Complete evaluation instructions
- **[Model READMEs](docs/models/)** - Individual architecture documentation
- **[CNN Improvements](docs/models/CNN_IMPROVEMENTS.md)** - CNN-specific enhancements
- **[MoE Architecture](docs/models/MOE_ARCHITECTURE.md)** - Mixture of Experts details

### Issues
- Check existing issues on GitHub
- Create detailed bug reports
- Include configuration files and logs
- Provide minimal reproducible examples

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VERDICT methodology developers
- Medical imaging research community
- PyTorch and scientific computing ecosystem
- Open source contributors

