import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet1d import EfficientNet1D

def test_efficientnet1d():
    """Test EfficientNet1D model"""
    print("Testing EfficientNet1D model...")
    
    # Typical VERDICT input/output dimensions
    input_dim = 100  # Example signal length
    output_dim = 4   # VERDICT parameters
    batch_size = 32
    
    # Create model
    model = EfficientNet1D(input_dim=input_dim, output_dim=output_dim)
    
    # Count parameters
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {output_dim})")
    
    # Verify output shape
    assert output.shape == (batch_size, output_dim), f"Wrong output shape: {output.shape}"
    
    print("✓ EfficientNet1D test passed!")
    print(f"✓ Parameter count: {param_count:,} (efficient for this task)")
    
    return True

if __name__ == "__main__":
    test_efficientnet1d()
