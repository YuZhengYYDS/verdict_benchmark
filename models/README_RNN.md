# Recurrent Neural Network (RNN) Regressor

## Overview

The RNN regressor adapts recurrent neural networks for VERDICT parameter prediction by reshaping flat feature vectors into sequences. It supports RNN, LSTM, and GRU architectures with configurable depth and sequence processing strategies.

## Architecture

### Key Components
- **Input Reshaping**: Converts flat features into sequences
- **Recurrent Layers**: RNN/LSTM/GRU cells with configurable depth
- **Sequence Processing**: Processes temporal patterns in reshaped data
- **Output Projection**: Maps final hidden state to target parameters

### Model Structure
```
Input (153) → Reshape (seq_len, feature_dim) → RNN Layers → Last Output → Linear (3)
```

### Sequence Reshaping Strategy
For input dimension 153:
- **Default**: Automatic sequence length calculation
- **Configured**: seq_len = 17, feature_dim = 9 (153 = 17 × 9)
- **Projection**: If not divisible, use linear projection to compatible size

## Configuration

### Model Parameters
- `hidden_dim`: RNN hidden state dimension (default: 128)
- `num_layers`: Number of RNN layers (default: 2)
- `rnn_type`: Type of RNN cell - 'RNN', 'LSTM', or 'GRU' (default: 'LSTM')
- `activation`: Activation function (default: 'tanh')
- `dropout`: Dropout rate between RNN layers (default: 0.1)
- `seq_len`: Sequence length for reshaping (default: 17)

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
python train.py --config configs/rnn.yaml
```

### Evaluation
```bash
python evaluate_models.py --config configs/rnn.yaml
```

## Architecture Details

### RNN Cell Types

#### 1. Vanilla RNN
- **Formula**: `h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)`
- **Pros**: Simple, fast
- **Cons**: Vanishing gradient problem

#### 2. LSTM (Long Short-Term Memory)
- **Components**: Input, forget, output gates + cell state
- **Pros**: Handles long-term dependencies, stable gradients
- **Cons**: More parameters, slower training

#### 3. GRU (Gated Recurrent Unit)
- **Components**: Reset and update gates
- **Pros**: Fewer parameters than LSTM, good performance
- **Cons**: Less expressive than LSTM

### Sequence Processing

#### Input Reshaping Logic
1. **Automatic Calculation**: Find optimal divisors of input_dim
2. **Preference**: Choose sequence length ≥ 8 for temporal modeling
3. **Fallback**: Use linear projection if input_dim not divisible

#### Example Reshaping
```python
# Input: (batch_size, 153)
# Reshape to: (batch_size, 17, 9)
# 153 = 17 × 9, creates meaningful sequence length
```

## Performance Characteristics

### Strengths
- **Temporal Modeling**: Can capture sequential patterns in data
- **Memory**: Maintains information across sequence steps
- **Flexibility**: Supports multiple RNN architectures
- **Robustness**: LSTM/GRU handle vanishing gradients well

### Limitations
- **Sequential Processing**: Cannot parallelize across time steps
- **Computational Cost**: More expensive than feedforward networks
- **Memory Requirements**: Needs to store hidden states
- **Artificial Sequences**: Medical data may not have natural sequence structure

## Mathematical Foundation

### LSTM Cell (Default)
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate values
C_t = f_t * C_{t-1} + i_t * C̃_t        # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate
h_t = o_t * tanh(C_t)                   # Hidden state
```

### Output Processing
- **Last Hidden State**: Use h_T from final time step
- **Activation**: Apply activation function to hidden state
- **Projection**: Linear layer maps to output dimensions

## Implementation Details

### Forward Pass
1. **Input Projection**: If needed, project input to compatible dimensions
2. **Reshaping**: Convert flat input to (batch, seq_len, feature_dim)
3. **RNN Processing**: Pass through RNN layers
4. **Output Selection**: Take final time step output
5. **Activation**: Apply activation function
6. **Projection**: Linear transformation to target dimensions

### Sequence Length Selection
```python
# Automatic sequence length calculation
divisors = [i for i in range(1, int(input_dim**0.5) + 1) if input_dim % i == 0]
seq_len = min([d for d in divisors if d >= 8] + [divisors[-1]])
```

## Hyperparameter Tuning

### Architecture Scaling
```yaml
# Larger hidden dimension
hidden_dim: 256

# Deeper network
num_layers: 4

# Different RNN type
rnn_type: GRU
```

### Sequence Configuration
```yaml
# Longer sequences
seq_len: 51  # 153 = 51 × 3

# Shorter sequences
seq_len: 9   # 153 = 9 × 17
```

### Regularization
```yaml
# Higher dropout for regularization
dropout: 0.2

# Different activation
activation: relu
```

## RNN Type Comparison

### When to Use Each Type

#### LSTM (Default)
- **Best for**: Long sequences, complex dependencies
- **Use when**: Maximum performance is needed
- **Avoid when**: Computational resources are limited

#### GRU
- **Best for**: Good balance of performance and efficiency
- **Use when**: Want fewer parameters than LSTM
- **Avoid when**: Need maximum expressiveness

#### Vanilla RNN
- **Best for**: Simple patterns, fast training
- **Use when**: Sequences are short (< 10 steps)
- **Avoid when**: Long sequences or complex patterns

## Expected Performance

### Typical Results
- **R² Score**: 0.83-0.90
- **RMSE**: 0.08-0.13
- **Training Time**: 20-35 minutes
- **Convergence**: May need more epochs than feedforward networks

### Performance by RNN Type
- **LSTM**: Highest accuracy, slowest training
- **GRU**: Good balance of accuracy and speed
- **RNN**: Fastest training, may underfit

## Best Practices

### Architecture Design
1. **Hidden Dimension**: Start with 128, increase if underfitting
2. **Layers**: 2-3 layers usually sufficient
3. **Sequence Length**: Choose based on input dimension factorization
4. **Dropout**: Use between layers for regularization

### Training Tips
1. **Gradient Clipping**: May help with training stability
2. **Learning Rate**: Start with 0.0003, may need adjustment
3. **Patience**: RNNs often need more epochs to converge
4. **Batch Size**: Smaller batches may work better

### Common Issues and Solutions
- **Vanishing Gradients**: Use LSTM/GRU instead of vanilla RNN
- **Exploding Gradients**: Apply gradient clipping
- **Slow Training**: Reduce hidden_dim or num_layers
- **Overfitting**: Increase dropout, reduce model complexity
- **Poor Performance**: Check if data has sequential structure

## Advanced Techniques

### Potential Improvements
- **Bidirectional RNN**: Process sequences in both directions
- **Attention Mechanisms**: Weight different time steps differently
- **Residual Connections**: Add skip connections in deep RNNs
- **Layer Normalization**: Normalize hidden states

### Alternative Architectures
- **Stacked RNNs**: Multiple RNN layers with different hidden dimensions
- **Encoder-Decoder**: Encode input sequence, decode to output
- **Sequence-to-Sequence**: Full sequence processing

## Limitations for Medical Data

### Considerations
- **Artificial Sequences**: Medical measurements may not have natural temporal order
- **Fixed Reshaping**: Same reshaping applied to all samples
- **No Temporal Semantics**: Time steps don't represent actual time
- **Computational Overhead**: May be overkill for tabular data

### When RNN May Not Be Optimal
- **Independent Features**: If features are independent
- **Small Datasets**: May overfit easily
- **Real-time Inference**: Slower than feedforward networks
- **Interpretability**: Less interpretable than simpler models

## File Structure
```
models/
├── rnn.py              # RNN implementation
├── mlp.py              # Shared activation functions
└── README_RNN.md       # This documentation

configs/
└── rnn.yaml           # Default configuration

checkpoints/
├── rnn_best.pt        # Best model weights
└── rnn_scaler.pkl     # Feature scaler
```

## References
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP.
- Chung, J., et al. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. NIPS Workshop.
