# Mixture of Experts (MoE) Models for VERDICT Parameter Prediction

## Overview

This document describes the Mixture of Experts (MoE) architectures implemented for the VERDICT benchmark. Two variants are provided to explore different approaches to expert routing and feature extraction.

## Architecture Details

### 1. MoERegressor (Standard Mixture of Experts)

The `MoERegressor` implements a standard MoE architecture with the following components:

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
- `top_k`: Number of experts to use per sample (default: 6)
- `noise_std`: Standard deviation for load balancing noise (default: 0.1)

### 2. AdaptiveMoERegressor (Hierarchical Mixture of Experts)

The `AdaptiveMoERegressor` implements a more sophisticated hierarchical approach:

#### Components:
- **Multi-scale Feature Extraction**: 1D convolutions to capture local signal patterns
- **Hierarchical Gating**: Two-level gating (coarse → fine) for better expert organization
- **Expert Groups**: 4 groups of 2 experts each, organized hierarchically

#### Key Features:
- **Coarse-to-Fine Selection**: First selects expert groups, then experts within groups
- **Multi-scale Features**: Combines global signal features with local convolutional features
- **Hierarchical Organization**: Better specialization through grouped experts

#### Architecture Flow:
1. Input → Multi-scale feature extraction (1D conv + global pooling)
2. Coarse gating → Select among 4 expert groups
3. Fine gating → Select experts within each group
4. Hierarchical combination → Final prediction

## Training Configuration

Both models use identical training configurations to ensure fair comparison:

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
```

## Why MoE for VERDICT?

1. **Signal Complexity**: VERDICT signals contain multiple biophysical components that may benefit from specialized processing
2. **Parameter Diversity**: The 8 VERDICT parameters represent different tissue properties that may require different modeling strategies
3. **Noise Robustness**: Multiple experts can provide redundancy and improve robustness to noise
4. **Scalability**: MoE architectures can handle large-scale problems while maintaining efficiency through sparse activation

## Usage

### Training Standard MoE:
```bash
python train.py --config configs/moe_regressor.yaml
```

### Training Adaptive MoE:
```bash
python train.py --config configs/adaptive_moe_regressor.yaml
```

## Expected Benefits

1. **Improved Accuracy**: Specialized experts for different signal patterns
2. **Better Generalization**: Reduced overfitting through ensemble-like behavior
3. **Interpretability**: Gating weights reveal which experts are active for different inputs
4. **Efficiency**: Sparse activation reduces computational cost during inference

## Monitoring

Both models return gating weights along with predictions, enabling analysis of:
- Expert utilization patterns
- Load balancing effectiveness
- Signal-specific expert preferences

Use W&B logs to monitor training progress and expert behavior patterns.

## File Structure

```
models/moe_regressor.py          # MoE model implementations
configs/moe_regressor.yaml       # Standard MoE configuration
configs/adaptive_moe_regressor.yaml  # Adaptive MoE configuration
test_moe.py                      # Model testing script
```

## Technical Notes

- Both models handle tuple outputs (prediction, gating_weights)
- Compatible with existing training infrastructure
- Layer normalization used for stable training
- Dropout regularization prevents overfitting
- Residual connections improve gradient flow
