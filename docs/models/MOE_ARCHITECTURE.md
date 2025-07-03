# Mixture of Experts (MoE) Model for VERDICT Parameter Prediction

## Overview

This document describes the Sparse Mixture of Experts (MoE) architecture implemented for the VERDICT benchmark. The sparse MoE approach uses selective expert activation to efficiently handle the complex signal-to-parameter mapping in VERDICT diffusion MRI.

## Architecture Details

### MoERegressor (Sparse Mixture of Experts)

The `MoERegressor` implements a sparse MoE architecture optimized for VERDICT parameter prediction with the following components:

#### Components:
- **Expert Networks**: 8 independent MLP networks, each specializing in different aspects of signal-to-parameter mapping
- **Gating Network**: A neural network that learns to weight expert contributions based on input signals
- **Input Processing**: Layer normalization and projection with residual connections
- **Output Processing**: Layer normalization for stable predictions

#### Key Features:
- **Sparse MoE**: Uses top-k expert selection (default: top 6 out of 8 experts) for efficiency
- **Load Balancing**: Adds noise to gating weights during training to prevent expert collapse
- **Residual Connections**: In input processing for better gradient flow

#### Parameters:
- `num_experts`: Number of expert networks (default: 8)
- `expert_hidden_dims`: Hidden dimensions for each expert (default: [128, 64])
- `gating_hidden_dim`: Hidden dimension for gating network (default: 64)
- `top_k`: Number of experts to use per sample (default: 6) - **Key sparse activation parameter**
- `noise_std`: Standard deviation for load balancing noise (default: 0.1)

## Why Sparse MoE for VERDICT?

The sparse MoE approach is particularly well-suited for VERDICT parameter prediction due to:

1. **Computational Efficiency**: Only activates top-k experts (6 out of 8), reducing computational cost
2. **Signal Complexity**: VERDICT signals contain multiple biophysical components that benefit from specialized processing
3. **Parameter Diversity**: The 8 VERDICT parameters represent different tissue properties requiring different modeling strategies
4. **Noise Robustness**: Multiple experts provide redundancy and improve robustness to noise
5. **Scalability**: Sparse activation enables handling of large-scale problems efficiently

## Training Configuration

The sparse MoE uses the following optimized configuration:

```yaml
lr: 0.0003
batch_size: 16
epochs: 400
early_stop_patience: 40
weight_decay: 0.0001

scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 15
  T_mult: 2
  eta_min: 0.000001
  warmup_epochs: 5

model:
  num_experts: 8
  expert_hidden_dims: [128, 64]
  gating_hidden_dim: 64
  top_k: 6  # Sparse activation: only 6 out of 8 experts
  noise_std: 0.1
  dropout: 0.1
```

## Usage

### Training Sparse MoE:
```bash
python train.py --config configs/moe_regressor.yaml
```

### Key Parameters:
- `num_experts: 8` - Total number of expert networks
- `top_k: 6` - Number of experts activated per sample (sparse activation)
- `expert_hidden_dims: [128, 64]` - Hidden layer dimensions for each expert
- `gating_hidden_dim: 64` - Gating network hidden dimension

## Expected Benefits

1. **Improved Accuracy**: Specialized experts for different signal patterns with sparse activation efficiency
2. **Better Generalization**: Reduced overfitting through ensemble-like behavior with controlled capacity
3. **Interpretability**: Gating weights reveal which experts are active for different inputs
4. **Computational Efficiency**: Sparse activation (top-k) reduces computational cost during inference
5. **Load Balancing**: Noise injection prevents expert collapse and ensures balanced utilization

## Monitoring

The sparse MoE model returns gating weights along with predictions, enabling analysis of:
- Expert utilization patterns and load balancing
- Sparse activation effectiveness (top-k selection)
- Signal-specific expert preferences
- Training stability through expert usage distribution

Use W&B logs to monitor training progress and expert behavior patterns.

## File Structure

```
models/moe_regressor.py          # Sparse MoE model implementation
configs/moe_regressor.yaml       # Sparse MoE configuration
```

## Technical Notes

- Model handles tuple outputs (prediction, gating_weights)
- Compatible with existing training infrastructure
- Layer normalization used for stable training
- Dropout regularization prevents overfitting
- Residual connections improve gradient flow
- Sparse activation via top-k expert selection for efficiency
- Load balancing noise prevents expert collapse
