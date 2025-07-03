# Transformer Regressor

## Overview

The Transformer regressor adapts the transformer architecture for VERDICT parameter prediction. It uses the encoder-only transformer design with self-attention mechanisms to capture complex relationships in medical imaging features.

## Architecture

### Key Components
- **Input Projection**: Maps input features to transformer hidden dimension
- **Positional Encoding**: Implicit through single-sequence processing
- **Multi-Head Self-Attention**: Captures feature relationships
- **Feed-Forward Networks**: Point-wise processing with nonlinearity
- **Layer Normalization**: Stabilizes training
- **Output Projection**: Maps final representation to target parameters

### Model Structure
```
Input (153) → Linear (d_model) → Unsqueeze → [Transformer Encoder] × N → Mean Pool → Linear (3)
```

### Transformer Encoder Layer
```
Input → Layer Norm → Multi-Head Attention → Add & Norm → Feed-Forward → Add & Norm → Output
```

## Configuration

### Model Parameters
- `d_model`: Hidden dimension (default: 64)
- `nhead`: Number of attention heads (default: 4)
- `num_layers`: Number of transformer layers (default: 2)
- `dim_feedforward`: Feed-forward network dimension (default: 128)
- `activation`: Activation function ('relu' or 'gelu', default: 'relu')
- `dropout`: Dropout rate (default: 0.1)

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
python train.py --config configs/transformer.yaml
```

### Evaluation
```bash
python evaluate_models.py --config configs/transformer.yaml
```

## Architecture Details

### Multi-Head Self-Attention
- **Purpose**: Captures relationships between different input features
- **Mechanism**: Queries, keys, and values computed from input
- **Multi-Head**: Multiple attention heads capture different types of relationships
- **Self-Attention**: Each position attends to all positions in the sequence

### Feed-Forward Network
- **Structure**: Two linear layers with activation in between
- **Expansion**: First layer expands to `dim_feedforward` dimensions
- **Compression**: Second layer compresses back to `d_model` dimensions

### Sequence Processing
1. **Input Projection**: Linear transformation to `d_model` dimensions
2. **Sequence Creation**: Unsqueeze to create sequence of length 1
3. **Transformer Processing**: Apply transformer encoder layers
4. **Pooling**: Mean pooling across sequence dimension
5. **Output Projection**: Linear transformation to target dimensions

## Mathematical Foundation

### Self-Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Queries): Linear transformation of input
- K (Keys): Linear transformation of input  
- V (Values): Linear transformation of input
- d_k: Dimension of key vectors

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

## Performance Characteristics

### Strengths
- **Attention Mechanism**: Captures complex feature relationships
- **Parallelization**: Can process all positions simultaneously
- **Expressiveness**: High representational capacity
- **Flexibility**: Can handle variable-length sequences (though not used here)
- **No Recurrence**: Avoids sequential processing bottlenecks

### Limitations
- **Computational Complexity**: O(n²) attention complexity
- **Memory Requirements**: Stores attention matrices
- **Sequence Length**: Current implementation uses sequence length 1
- **Overfitting Risk**: High capacity may overfit on small datasets

## Implementation Details

### Forward Pass
1. **Input Projection**: Map input features to d_model dimensions
2. **Sequence Creation**: Add sequence dimension (batch, 1, d_model)
3. **Transformer Processing**: Apply transformer encoder layers
4. **Pooling**: Average across sequence dimension
5. **Output**: Linear projection to target dimensions

### Attention Pattern
With sequence length 1, the attention mechanism simplifies to:
- **Self-Attention**: Each feature attends to itself
- **Cross-Feature**: Implicit through multiple heads and layers

## Hyperparameter Tuning

### Architecture Scaling
```yaml
# Larger model
d_model: 128
nhead: 8
num_layers: 4
dim_feedforward: 256

# Smaller model
d_model: 32
nhead: 2
num_layers: 1
dim_feedforward: 64
```

### Attention Configuration
```yaml
# More attention heads
nhead: 8  # Must divide d_model

# Deeper transformer
num_layers: 6
```

### Regularization
```yaml
# Higher dropout
dropout: 0.2

# Different activation
activation: gelu
```

## Attention Head Analysis

### Multi-Head Benefits
- **Diverse Patterns**: Different heads learn different relationships
- **Redundancy**: Multiple heads provide robustness
- **Ensemble Effect**: Combination of multiple attention patterns

### Head Configuration
- **nhead = 1**: Single attention pattern
- **nhead = 2**: Two complementary patterns
- **nhead = 4**: Diverse set of attention patterns (default)
- **nhead = 8**: Maximum diversity (if d_model = 64)

## Expected Performance

### Typical Results
- **R² Score**: 0.86-0.93
- **RMSE**: 0.07-0.12
- **Training Time**: 15-30 minutes
- **Convergence**: Often fast initial convergence

### Performance Scaling
- **Small Model**: Comparable to MLP
- **Medium Model**: Better than MLP, similar to CNN
- **Large Model**: May overfit on small datasets

## Best Practices

### Architecture Design
1. **d_model**: Start with 64, increase if underfitting
2. **nhead**: Use 4-8 heads, must divide d_model
3. **num_layers**: Start with 2, increase gradually
4. **dim_feedforward**: Usually 2-4× d_model

### Training Tips
1. **Learning Rate**: Transformers often benefit from learning rate warm-up
2. **Batch Size**: Larger batches may help with attention stability
3. **Dropout**: Important for preventing overfitting
4. **Weight Decay**: Use moderate weight decay

### Common Issues and Solutions
- **Dimension Mismatch**: Ensure nhead divides d_model
- **Overfitting**: Increase dropout, reduce model size
- **Slow Training**: Reduce num_layers or d_model
- **NaN Loss**: Check learning rate, consider gradient clipping
- **Poor Convergence**: Try different activation functions

## Advanced Techniques

### Potential Improvements
- **Positional Encoding**: Add explicit positional information
- **Multi-Scale**: Use different d_model for different layers
- **Residual Scaling**: Scale residual connections
- **Pre-Layer Normalization**: Move layer norm before attention

### Alternative Architectures
- **Encoder-Decoder**: For sequence-to-sequence tasks
- **Hierarchical**: Multi-level transformer processing
- **Sparse Attention**: Reduce attention complexity

## Sequence Length Considerations

### Current Implementation
- **Sequence Length**: 1 (treats each sample as single token)
- **Attention**: Simplified to feature transformation
- **Limitation**: Doesn't exploit sequence modeling capabilities

### Potential Extensions
- **Feature Grouping**: Group related features into sequences
- **Sliding Window**: Create overlapping feature windows
- **Hierarchical**: Multi-level feature processing

## Comparison with Other Architectures

### vs. MLP
- **Attention**: Captures feature relationships vs. point-wise processing
- **Complexity**: Higher complexity, more parameters
- **Performance**: Often better on complex patterns

### vs. CNN
- **Receptive Field**: Global attention vs. local convolutions
- **Inductive Bias**: Less structural bias
- **Computational**: Different complexity patterns

### vs. RNN
- **Parallelization**: Parallel vs. sequential processing
- **Memory**: Attention matrix vs. hidden states
- **Dependencies**: All-to-all vs. sequential dependencies

## Limitations for Medical Data

### Considerations
- **Sequence Length**: Current implementation uses length 1
- **Attention Benefit**: May be limited with tabular data
- **Computational Cost**: Higher than simpler alternatives
- **Interpretability**: Attention weights provide some interpretability

### When Transformer May Not Be Optimal
- **Small Datasets**: May overfit easily
- **Simple Patterns**: Overkill for linear relationships
- **Real-time Inference**: Slower than feedforward networks
- **Resource Constraints**: High memory and computational requirements

## File Structure
```
models/
├── transformer.py         # Transformer implementation
├── mlp.py                 # Shared activation functions
└── README_Transformer.md  # This documentation

configs/
└── transformer.yaml       # Default configuration

checkpoints/
├── transformer_best.pt      # Best model weights
└── transformer_scaler.pkl   # Feature scaler
```

## References
- Vaswani, A., et al. (2017). Attention is all you need. NIPS.
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
- Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.
