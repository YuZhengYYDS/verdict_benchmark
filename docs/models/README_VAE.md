# Variational Autoencoder (VAE) Regressor

## Overview

The VAE regressor combines variational autoencoders with regression to predict VERDICT parameters. It learns a latent representation of the input data through an encoder-decoder architecture while simultaneously predicting target parameters from the latent space.

## Architecture

### Key Components
- **Encoder**: Maps input to latent space distribution (μ, σ)
- **Latent Space**: Probabilistic latent representation
- **Decoder**: Reconstructs input from latent representation
- **Regressor**: Predicts target parameters from latent space
- **Variational Loss**: Combines reconstruction, regression, and KL divergence

### Model Structure
```
Input (153) → Encoder → μ, log_σ → Reparameterization → Latent (32) → Decoder → Reconstruction (153)
                                                         ↓
                                                    Regressor → Output (3)
```

### Loss Function
```
Total Loss = Regression Loss + α × Reconstruction Loss + β × KL Divergence
```

## Configuration

### Model Parameters
- `latent_dim`: Latent space dimension (default: 32)
- `hidden_dims`: Encoder/decoder hidden dimensions (default: [128, 64])
- `activation`: Activation function (default: 'relu')
- `dropout`: Dropout rate (default: 0.1)
- `beta`: KL divergence weight (default: 0.1)
- `alpha`: Reconstruction loss weight (default: 0.5)

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
python train.py --config configs/vae_regressor.yaml
```

### Evaluation
```bash
python evaluate_models.py --config configs/vae_regressor.yaml
```

## Architecture Details

### Encoder Network
```
Input → Hidden Layers → μ (mean), log_σ (log variance)
```
- **Purpose**: Maps input to latent distribution parameters
- **Output**: Mean and log variance of latent distribution
- **Architecture**: Fully connected layers with activation and dropout

### Reparameterization Trick
```
z = μ + σ × ε, where ε ~ N(0, I)
```
- **Purpose**: Enables backpropagation through stochastic sampling
- **Implementation**: `z = mu + torch.exp(0.5 * logvar) * eps`
- **Benefit**: Maintains gradient flow while sampling

### Decoder Network
```
Latent → Hidden Layers → Reconstruction
```
- **Purpose**: Reconstructs input from latent representation
- **Architecture**: Reverse of encoder (symmetric)
- **Output**: Reconstructed input features

### Regression Head
```
Latent → Hidden Layer → Dropout → Output
```
- **Purpose**: Predicts target parameters from latent space
- **Input**: Sampled latent representation
- **Output**: VERDICT parameters

## Mathematical Foundation

### Variational Lower Bound
The VAE maximizes the Evidence Lower Bound (ELBO):
```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```

### KL Divergence (Regularization)
```
KL = -0.5 × Σ(1 + log_σ² - μ² - σ²)
```

### Combined Loss Function
```
L = MSE(y_pred, y_true) + α × MSE(x_recon, x) + β × KL
```

Where:
- **MSE(y_pred, y_true)**: Regression loss
- **MSE(x_recon, x)**: Reconstruction loss
- **KL**: Kullback-Leibler divergence

## Performance Characteristics

### Strengths
- **Latent Representation**: Learns meaningful compressed representations
- **Regularization**: KL divergence provides implicit regularization
- **Uncertainty**: Probabilistic latent space captures uncertainty
- **Generative**: Can generate new samples from latent space
- **Disentanglement**: May learn disentangled representations

### Limitations
- **Complexity**: More complex than discriminative models
- **Training Difficulty**: Requires balancing multiple loss terms
- **Computational Cost**: Higher than standard regression models
- **Hyperparameter Sensitivity**: Sensitive to loss weights (α, β)

## Implementation Details

### Forward Pass
1. **Encoding**: Compute μ and log_σ from input
2. **Sampling**: Sample latent vector using reparameterization
3. **Decoding**: Reconstruct input from latent vector
4. **Regression**: Predict parameters from latent vector
5. **Loss**: Compute combined loss

### Training Process
1. **Forward Pass**: Get predictions and reconstructions
2. **Loss Computation**: Combine all loss terms
3. **Backpropagation**: Compute gradients
4. **Optimization**: Update parameters

## Hyperparameter Tuning

### Architecture Scaling
```yaml
# Larger latent space
latent_dim: 64

# Deeper encoder/decoder
hidden_dims: [256, 128, 64]

# Wider networks
hidden_dims: [256, 256]
```

### Loss Weight Tuning
```yaml
# Higher reconstruction weight
alpha: 1.0

# Higher KL weight (more regularization)
beta: 0.5

# Lower KL weight (less regularization)
beta: 0.01
```

### Regularization
```yaml
# Higher dropout
dropout: 0.2

# Different activation
activation: gelu
```

## Loss Weight Analysis

### Beta (KL Weight)
- **High β (>1.0)**: Strong regularization, may underfit
- **Medium β (0.1-1.0)**: Good balance (default: 0.1)
- **Low β (<0.1)**: Weak regularization, may overfit
- **β = 0**: Reduces to standard autoencoder

### Alpha (Reconstruction Weight)
- **High α (>1.0)**: Focus on reconstruction quality
- **Medium α (0.1-1.0)**: Balanced approach (default: 0.5)
- **Low α (<0.1)**: Focus on regression performance
- **α = 0**: Ignores reconstruction (not recommended)

## Expected Performance

### Typical Results
- **R² Score**: 0.85-0.92
- **RMSE**: 0.08-0.12
- **Training Time**: 25-40 minutes
- **Convergence**: May require more epochs than discriminative models

### Performance Factors
- **Latent Dimension**: Larger dimension may improve performance
- **Architecture Depth**: Deeper networks may capture more complex patterns
- **Loss Weights**: Proper tuning crucial for good performance

## Best Practices

### Architecture Design
1. **Latent Dimension**: Start with 32, increase if underfitting
2. **Hidden Layers**: Use symmetric encoder-decoder architecture
3. **Activation**: ReLU works well, GELU may provide improvements
4. **Dropout**: Important for preventing overfitting

### Loss Weight Tuning
1. **Start Conservative**: Begin with small β (0.01-0.1)
2. **Monitor Components**: Track each loss component separately
3. **Gradual Increase**: Increase β if model overfits
4. **Balance**: Aim for comparable magnitudes of loss components

### Training Tips
1. **Learning Rate**: May need lower learning rate than discriminative models
2. **Warm-up**: Consider KL warm-up (gradually increase β)
3. **Monitoring**: Watch reconstruction quality alongside regression performance
4. **Patience**: VAEs often need more epochs to converge

### Common Issues and Solutions
- **Posterior Collapse**: Reduce β, increase architecture capacity
- **Poor Reconstruction**: Increase α, check decoder architecture
- **Overfitting**: Increase β, add more dropout
- **Slow Convergence**: Reduce learning rate, increase patience
- **NaN Loss**: Check loss weights, gradient clipping may help

## Advanced Techniques

### Potential Improvements
- **β-VAE**: Controllable disentanglement with β scheduling
- **WAE**: Wasserstein Autoencoder for better training stability
- **Conditional VAE**: Condition on auxiliary information
- **Hierarchical VAE**: Multi-level latent representations

### Training Strategies
- **KL Annealing**: Gradually increase β during training
- **Cyclical Annealing**: Periodic β scheduling
- **Free Bits**: Minimum KL divergence per latent dimension
- **Spectral Normalization**: Stabilize training dynamics

## Latent Space Analysis

### Interpretation
- **Dimensionality**: Each dimension may capture different aspects
- **Interpolation**: Smooth transitions between samples
- **Clustering**: Similar samples may cluster in latent space
- **Disentanglement**: Different dimensions may control different factors

### Visualization
```python
# Sample from latent space
z = torch.randn(100, latent_dim)
reconstructed = model.decode(z)

# Latent space interpolation
z1, z2 = latent_codes[0], latent_codes[1]
interpolated = torch.lerp(z1, z2, torch.linspace(0, 1, 10))
```

## Comparison with Other Models

### vs. Standard Regression
- **Representation**: Learns latent representations vs. direct mapping
- **Regularization**: KL divergence vs. weight decay
- **Interpretability**: Latent space provides insights

### vs. Autoencoder
- **Probabilistic**: Probabilistic latent space vs. deterministic
- **Regularization**: KL divergence prevents overfitting
- **Uncertainty**: Captures uncertainty in representations

### vs. GAN
- **Training**: More stable training than GANs
- **Likelihood**: Provides likelihood estimates
- **Mode Collapse**: Less prone to mode collapse

## Applications Beyond Regression

### Potential Uses
- **Data Augmentation**: Generate synthetic training samples
- **Anomaly Detection**: Identify outliers using reconstruction error
- **Dimensionality Reduction**: Visualize high-dimensional data
- **Feature Learning**: Extract meaningful features for other tasks

## File Structure
```
models/
├── vae_regressor.py         # VAE regressor implementation
├── mlp.py                   # Shared activation functions
└── README_VAE.md            # This documentation

configs/
└── vae_regressor.yaml       # Default configuration

checkpoints/
├── vae_regressor_best.pt      # Best model weights
└── vae_regressor_scaler.pkl   # Feature scaler
```

## References
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. ICLR.
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. ICML.
- Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. ICLR.
- Burgess, C. P., et al. (2018). Understanding disentangling in β-VAE. NIPS.
