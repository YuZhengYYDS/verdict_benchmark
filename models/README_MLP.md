# Multi-Layer Perceptron (MLP) Regressor

## Overview

The MLP (Multi-Layer Perceptron) regressor is a simple feedforward neural network designed for VERDICT parameter prediction. It consists of fully connected layers with configurable depth, width, and activation functions.

## Architecture

### Key Components
- **Input Layer**: Accepts flattened feature vectors
- **Hidden Layers**: Multiple fully connected layers with configurable dimensions
- **Activation Functions**: Support for ReLU, Tanh, Sigmoid, Swish, GELU, and Leaky ReLU
- **Output Layer**: Linear layer producing VERDICT parameters

### Model Structure
```
Input (153 features) → Hidden Layer 1 (150) → Hidden Layer 2 (150) → Hidden Layer 3 (150) → Output (3 parameters)
```

## Configuration

### Model Parameters
- `hidden_dims`: List of hidden layer dimensions (default: [150, 150, 150])
- `activation`: Activation function name (default: 'relu')

### Training Parameters
- `epochs`: 400
- `batch_size`: 16
- `lr`: 0.0003
- `weight_decay`: 0.0001
- `early_stop_patience`: 40

### Scheduler
- **Type**: CosineAnnealingWarmRestarts
- **T_0**: 15 epochs
- **T_mult**: 2
- **eta_min**: 1e-6
- **warmup_epochs**: 5

## Usage

### Training
```bash
python train.py --config configs/mlp.yaml
```

### Evaluation
```bash
python evaluate_models.py --config configs/mlp.yaml
```

## Architecture Details

### Supported Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Default choice for stability
- **Tanh**: `f(x) = tanh(x)` - Bounded output, good for gradients
- **Sigmoid**: `f(x) = 1/(1+e^(-x))` - Bounded between 0 and 1
- **Swish**: `f(x) = x * sigmoid(x)` - Smooth, non-monotonic
- **GELU**: Gaussian Error Linear Unit - Smooth approximation of ReLU
- **Leaky ReLU**: `f(x) = max(0.1x, x)` - Prevents dead neurons

### Layer Configuration
The default configuration uses three hidden layers of 150 neurons each:
- **Layer 1**: 153 → 150 with ReLU activation
- **Layer 2**: 150 → 150 with ReLU activation  
- **Layer 3**: 150 → 150 with ReLU activation
- **Output**: 150 → 3 (linear)

## Performance Characteristics

### Strengths
- **Simplicity**: Easy to understand and implement
- **Flexibility**: Configurable depth and width
- **Stability**: Well-understood training dynamics
- **Fast Training**: Quick convergence on tabular data
- **Memory Efficient**: Low memory footprint

### Limitations
- **Limited Expressiveness**: May struggle with complex non-linear patterns
- **No Inductive Bias**: Lacks structural assumptions about data
- **Vanishing Gradients**: Can occur with very deep networks
- **Overfitting Risk**: May memorize training data without regularization

## Hyperparameter Tuning

### Architecture Tuning
```yaml
# Deeper network
hidden_dims: [200, 150, 100, 50]

# Wider network  
hidden_dims: [300, 300, 300]

# Pyramid structure
hidden_dims: [256, 128, 64, 32]
```

### Activation Function Selection
```yaml
# For smooth gradients
activation: gelu

# For bounded outputs
activation: tanh

# For avoiding dead neurons
activation: leaky_relu
```

### Regularization Options
```yaml
# Increase weight decay
weight_decay: 0.001

# Reduce learning rate
lr: 0.0001

# Earlier stopping
early_stop_patience: 20
```

## Implementation Details

### Forward Pass
1. Input features are passed through each hidden layer
2. Each layer applies linear transformation followed by activation
3. Final layer produces raw parameter predictions

### Training Process
1. **Initialization**: Xavier/Kaiming initialization for weights
2. **Forward Pass**: Compute predictions
3. **Loss Calculation**: MSE loss between predictions and targets
4. **Backward Pass**: Compute gradients via backpropagation
5. **Optimization**: Adam optimizer with weight decay
6. **Scheduling**: Cosine annealing with warm restarts

## Expected Performance

### Typical Results
- **R² Score**: 0.85-0.92
- **RMSE**: 0.08-0.12
- **Training Time**: 10-20 minutes
- **Inference Speed**: Very fast (~1ms per batch)

### Comparison with Other Models
- **vs CNN**: Simpler but may have lower accuracy
- **vs RNN**: Faster training, no sequence modeling
- **vs Transformer**: Much faster, less powerful
- **vs VAE**: Direct mapping, no latent space

## Best Practices

### For Better Performance
1. **Data Normalization**: Always normalize input features
2. **Learning Rate**: Start with 0.0003 and adjust based on loss curves
3. **Architecture**: Start with 3 layers, increase if underfitting
4. **Regularization**: Use weight decay to prevent overfitting
5. **Early Stopping**: Monitor validation loss for optimal stopping

### Common Issues
- **Vanishing Gradients**: Use ReLU or GELU activation
- **Exploding Gradients**: Reduce learning rate or add gradient clipping
- **Overfitting**: Increase weight decay or reduce model complexity
- **Slow Convergence**: Increase learning rate or use learning rate scheduling

## File Structure
```
models/
├── mlp.py              # MLP implementation
└── README_MLP.md       # This documentation

configs/
└── mlp.yaml           # Default configuration

checkpoints/
├── mlp_best.pt        # Best model weights
└── mlp_scaler.pkl     # Feature scaler
```

## References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
