# TabNet Regressor for VERDICT Parameter Prediction

## 🎯 Overview

TabNet (Tabular Network) is a state-of-the-art deep learning architecture specifically designed for tabular data. It combines the power of attention mechanisms with interpretability, making it particularly suitable for medical parameter prediction tasks like VERDICT.

## 🏗️ Architecture

### Key Components

1. **Sequential Attention**: Uses learnable masks to select relevant features at each decision step
2. **Ghost Batch Normalization**: Applies batch normalization on smaller virtual batches for better performance
3. **Gated Linear Units (GLU)**: Enables non-linear feature transformations with gating mechanisms
4. **Feature Reusage**: Balances feature selection across decision steps with a reusage coefficient

### Architecture Flow

```
Input Features (153-dim)
    ↓
Initial Batch Normalization
    ↓
┌─────────────────────────────────┐
│        Decision Step 1          │
│  ┌─────────────────────────────┐│
│  │   Attentive Transformer     ││  → Attention Mask 1
│  └─────────────────────────────┘│
│              ↓                  │
│  ┌─────────────────────────────┐│
│  │   Feature Transformer       ││  → Decision Output 1
│  └─────────────────────────────┘│
└─────────────────────────────────┘
    ↓ (Updated Prior)
┌─────────────────────────────────┐
│        Decision Step 2          │
│            ...                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│        Decision Step N          │
└─────────────────────────────────┘
    ↓
Sum All Decision Outputs
    ↓
Final Linear Mapping
    ↓
VERDICT Parameters (8-dim)
```

## 🔧 Configuration Parameters

### Core Architecture Parameters

- **`n_d`** (default: 16): Width of the decision prediction layer
  - Controls the capacity of each decision step
  - Typical range: 8-64
  - Higher values = more complex decisions per step

- **`n_a`** (default: 16): Width of the attention embedding
  - Controls the complexity of feature selection
  - Usually set equal to `n_d`
  - Higher values = more sophisticated attention

- **`n_steps`** (default: 4): Number of decision steps
  - More steps = more sequential feature selection
  - Typical range: 3-10
  - Balance between performance and interpretability

### Feature Selection Parameters

- **`gamma`** (default: 1.5): Feature reusage coefficient
  - Controls how much features can be reused across steps
  - Values > 1.0 encourage diverse feature usage
  - Range: 1.0-2.0

### Training Parameters

- **`n_independent`** (default: 2): Independent GLU layers per step
  - Specific to each decision step
  - More layers = higher capacity per step

- **`n_shared`** (default: 2): Shared GLU layers across steps
  - Shared feature transformations
  - Promotes consistency across steps

### Regularization Parameters

- **`virtual_batch_size`** (default: 128): Ghost batch norm size
  - Smaller than actual batch size
  - Helps with generalization
  - Typical range: 64-256

- **`momentum`** (default: 0.02): Batch normalization momentum
  - Lower than standard BN momentum
  - Helps with stability in ghost BN

## 🚀 Quick Start

### Basic Training
```bash
python train.py --config configs/tabnet_regressor.yaml
```

### Advanced Training with Custom Parameters
```yaml
model:
  type: tabnet_regressor
  class_name: TabNetRegressor
  params:
    n_d: 32                    # Larger decision layer
    n_a: 32                    # Larger attention
    n_steps: 6                 # More decision steps
    gamma: 1.8                 # More diverse feature usage
    virtual_batch_size: 64     # Smaller virtual batches
```

## 📊 Performance Characteristics

### Strengths
- **Interpretability**: Built-in feature importance via attention masks
- **Tabular Optimization**: Specifically designed for structured data
- **Feature Selection**: Automatic relevance-based feature selection
- **Competitive Performance**: Often matches or exceeds gradient boosting

### Training Behavior
- **Convergence**: May require more epochs than simpler models
- **Memory Usage**: Higher than basic MLPs due to attention mechanisms
- **Batch Size**: Can handle larger batches efficiently
- **Learning Rate**: Often works well with slightly higher learning rates

## 🔍 Interpretability Features

### Feature Importance Analysis

```python
from models.tabnet_regressor import TabNetRegressor

# Load trained model
model = TabNetRegressor(input_dim=153, output_dim=8, ...)
model.load_state_dict(torch.load('checkpoints/tabnet_best.pt'))

# Get feature importance for a batch
with torch.no_grad():
    importance = model.get_feature_importance(input_batch)
    # Shape: (batch_size, input_dim)

# Get global feature importance
global_importance = model.get_global_feature_importance(dataloader)
# Shape: (input_dim,)
```

### Attention Masks Visualization

```python
# Forward pass with masks
output, masks = model.forward_masks(input_batch)
# masks shape: (batch_size, n_steps, input_dim)

# Visualize attention patterns
import matplotlib.pyplot as plt
import seaborn as sns

# Average across batch for visualization
avg_masks = masks.mean(dim=0).cpu().numpy()  # (n_steps, input_dim)

plt.figure(figsize=(12, 6))
sns.heatmap(avg_masks, cmap='YlOrRd', cbar=True)
plt.xlabel('Input Features')
plt.ylabel('Decision Steps')
plt.title('TabNet Attention Masks - Feature Selection Pattern')
plt.show()
```

## 🎛️ Hyperparameter Tuning

### Performance Tuning Guide

1. **Start with defaults**: Use the provided configuration as baseline
2. **Adjust capacity**: Increase `n_d` and `n_a` for complex datasets
3. **Tune steps**: More `n_steps` for datasets with many features
4. **Feature reusage**: Adjust `gamma` based on feature diversity needs
5. **Regularization**: Tune `virtual_batch_size` and `momentum` for stability

### Common Configurations

**Small Model (Fast Training)**:
```yaml
n_d: 8
n_a: 8
n_steps: 3
gamma: 1.3
```

**Medium Model (Balanced)**:
```yaml
n_d: 16
n_a: 16
n_steps: 4
gamma: 1.5
```

**Large Model (Maximum Performance)**:
```yaml
n_d: 32
n_a: 32
n_steps: 6
gamma: 1.8
```

## 🔬 Medical Applications

### VERDICT Parameter Prediction

TabNet is particularly well-suited for VERDICT parameter prediction because:

1. **Feature Interpretability**: Can identify which imaging features are most important for each parameter
2. **Sequential Processing**: Mimics clinical decision-making with step-by-step feature evaluation
3. **Attention Mechanisms**: Similar to how radiologists focus on relevant image regions
4. **Robust to Noise**: Ghost batch normalization helps with noisy medical data

### Clinical Insights

The attention masks can provide clinical insights:
- **Vascular Parameter**: May focus on perfusion-related features
- **Extracellular Parameter**: May emphasize diffusion tensor features  
- **Restricted Parameter**: May highlight cellular density indicators

## 📈 Expected Performance

Based on similar tabular regression tasks:

- **R² Score**: 0.85-0.92 (competitive with ensemble methods)
- **RMSE**: 0.05-0.08 (potentially lower than simpler architectures)
- **Training Time**: 30-60 minutes (longer than MLPs, shorter than transformers)
- **Parameters**: ~250K (moderate complexity)

## 🛠️ Troubleshooting

### Common Issues

**Memory Errors**:
- Reduce `virtual_batch_size`
- Decrease `batch_size`
- Lower `n_d` and `n_a`

**Slow Convergence**:
- Increase learning rate
- Adjust `gamma` for better feature selection
- Check feature scaling

**Poor Performance**:
- Increase `n_steps` for more complex feature selection
- Adjust `n_d` and `n_a` for higher capacity
- Ensure proper data preprocessing

### Performance Tips

- **GPU Acceleration**: TabNet benefits significantly from GPU training
- **Mixed Precision**: Can use automatic mixed precision for memory efficiency
- **Gradient Clipping**: May help with training stability
- **Early Stopping**: Use validation loss with patience for optimal stopping

## 📚 References

1. **Original Paper**: Sercan Ö. Arık, Tomas Pfister. "TabNet: Attentive Interpretable Tabular Learning." AAAI 2021.
2. **Implementation**: Based on the official TabNet paper with adaptations for regression
3. **Medical Applications**: Adapted for medical parameter prediction tasks

## 🔗 Related Models

- **MLP**: Simpler baseline for comparison
- **Transformer**: Alternative attention-based architecture
- **Gradient Boosting**: Traditional tabular learning baseline
- **Deep FM**: Another tabular-specific deep learning approach
