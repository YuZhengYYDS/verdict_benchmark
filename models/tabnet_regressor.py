import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization for TabNet.
    Applies batch normalization on smaller virtual batches.
    """
    def __init__(self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)


class GLU(nn.Module):
    """
    Gated Linear Unit activation function.
    """
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        out, gate = x.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class FeatureTransformer(nn.Module):
    """
    Feature Transformer block for TabNet.
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 n_independent: int = 2, n_shared: int = 2, virtual_batch_size: int = 128):
        super().__init__()
        
        # Shared layers
        self.shared = nn.ModuleList()
        current_dim = input_dim
        for i in range(n_shared):
            self.shared.append(GLU(current_dim, output_dim, virtual_batch_size))
            current_dim = output_dim
        
        # Independent layers
        self.specifics = nn.ModuleList()
        for i in range(n_independent):
            self.specifics.append(GLU(current_dim, output_dim, virtual_batch_size))
        
        self.scale = torch.sqrt(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply shared layers
        for layer in self.shared:
            x = layer(x)
        
        # Apply independent layers with residual connections
        for layer in self.specifics:
            x = self.scale * x + layer(x)
        
        return x


class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer for TabNet feature selection.
    """
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = 128):
        super().__init__()
        self.fc = nn.Linear(output_dim, input_dim, bias=False)  # Changed: output_dim -> input_dim
        self.bn = GhostBatchNorm(input_dim, virtual_batch_size)  # Changed: output_dim -> input_dim

    def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = priors * x
        return torch.softmax(x, dim=-1)


class TabNetRegressor(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning
    
    Paper: https://arxiv.org/abs/1908.07442
    
    This is a PyTorch implementation of TabNet specifically designed for regression tasks.
    TabNet uses sequential attention to select relevant features at each step and has
    built-in interpretability through attention masks.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output targets
        n_d: Width of the decision prediction layer (default: 8)
        n_a: Width of the attention embedding for each mask (default: 8)  
        n_steps: Number of decision steps (default: 3)
        gamma: Coefficient for feature reusage in the masks (default: 1.3)
        n_independent: Number of independent Gated Linear Units at each step (default: 2)
        n_shared: Number of shared Gated Linear Units at each step (default: 2)
        epsilon: Avoid log(0) in entropy regularization (default: 1e-15)
        virtual_batch_size: Batch size for Ghost Batch Normalization (default: 128)
        momentum: Momentum for Ghost Batch Normalization (default: 0.02)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 epsilon: float = 1e-15,
                 virtual_batch_size: int = 128,
                 momentum: float = 0.02):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        
        # Input batch normalization
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size, momentum)
        
        # Decision steps
        self.feat_transformers = nn.ModuleList([
            FeatureTransformer(input_dim, n_d + n_a, n_independent, n_shared, virtual_batch_size)
            for _ in range(n_steps)
        ])
        
        self.att_transformers = nn.ModuleList([
            AttentiveTransformer(input_dim, n_a, virtual_batch_size)
            for _ in range(n_steps)
        ])
        
        # Final mapping for regression
        self.final_mapping = nn.Linear(n_d, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TabNet.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        res = 0
        x = self.initial_bn(x)
        prior = torch.ones(x.shape).to(x.device)
        
        for step in range(self.n_steps):
            # Apply feature transformer to get processed features
            out = self.feat_transformers[step](x)
            d = F.relu(out[:, :self.n_d])
            a = out[:, self.n_d:]
            
            # Generate attention mask from attention features
            M = self.att_transformers[step](prior, a)
            
            # Apply mask to input for next iteration
            x = torch.mul(M, x)
            
            # Update prior for next step
            prior = torch.mul(self.gamma - M, prior)
            
            # Accumulate decision
            res = res + d
        
        # Final regression output
        res = self.final_mapping(res)
        return res

    def forward_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention masks for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, masks) where:
            - output: Regression output of shape (batch_size, output_dim)
            - masks: Attention masks of shape (batch_size, n_steps, input_dim)
        """
        res = 0
        x = self.initial_bn(x)
        prior = torch.ones(x.shape).to(x.device)
        masks = []
        
        for step in range(self.n_steps):
            # Apply feature transformer to get processed features
            out = self.feat_transformers[step](x)
            d = F.relu(out[:, :self.n_d])
            a = out[:, self.n_d:]
            
            # Generate attention mask from attention features
            M = self.att_transformers[step](prior, a)
            masks.append(M)
            
            # Apply mask to input for next iteration
            x = torch.mul(M, x)
            
            # Update prior for next step
            prior = torch.mul(self.gamma - M, prior)
            
            # Accumulate decision
            res = res + d
        
        # Final regression output
        res = self.final_mapping(res)
        masks = torch.stack(masks, dim=1)
        
        return res, masks

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores for a batch of inputs.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Feature importance scores of shape (batch_size, input_dim)
        """
        _, masks = self.forward_masks(x)
        # Sum importance across all decision steps
        importance = torch.sum(masks, dim=1)
        return importance

    def get_global_feature_importance(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Get global feature importance averaged across the entire dataset.
        
        Args:
            dataloader: DataLoader containing the dataset
            
        Returns:
            Global feature importance of shape (input_dim,)
        """
        self.eval()
        total_importance = None
        total_samples = 0
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(next(self.parameters()).device)
                batch_importance = self.get_feature_importance(batch_x)
                
                if total_importance is None:
                    total_importance = torch.sum(batch_importance, dim=0)
                else:
                    total_importance += torch.sum(batch_importance, dim=0)
                
                total_samples += batch_x.shape[0]
        
        # Average across all samples
        global_importance = total_importance / total_samples
        return global_importance
