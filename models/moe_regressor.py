import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import get_activation


class Expert(nn.Module):
    """Single expert network in the MoE architecture."""
    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu', dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GatingNetwork(nn.Module):
    """Gating network that determines expert weights."""
    def __init__(self, input_dim, num_experts, hidden_dim=64, activation='relu', dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, x):
        # Return softmax probabilities for expert selection
        return F.softmax(self.network(x), dim=-1)


class MoERegressor(nn.Module):
    """
    Mixture of Experts (MoE) Regressor for VERDICT parameter prediction.
    
    This architecture uses multiple expert networks, each specializing in different 
    aspects of the signal-to-parameter mapping, with a gating network that learns
    to weight the contributions of each expert based on the input.
    
    Args:
        input_dim (int): Input feature dimension (153 for VERDICT signals)
        output_dim (int): Output dimension (8 for VERDICT parameters)
        num_experts (int): Number of expert networks
        expert_hidden_dims (list): Hidden dimensions for each expert
        gating_hidden_dim (int): Hidden dimension for gating network
        activation (str): Activation function ('relu', 'gelu', 'swish', etc.)
        dropout (float): Dropout rate
        noise_std (float): Standard deviation for load balancing noise
        top_k (int): Number of top experts to use (if < num_experts, sparse MoE)
    """
    def __init__(self, input_dim, output_dim, num_experts=8, 
                 expert_hidden_dims=[128, 64], gating_hidden_dim=64,
                 activation='relu', dropout=0.1, noise_std=0.1, top_k=None):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else num_experts
        self.noise_std = noise_std
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, expert_hidden_dims, activation, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating_network = GatingNetwork(
            input_dim, num_experts, gating_hidden_dim, activation, dropout
        )
        
        # Input preprocessing layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # Output post-processing
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Preprocess input
        x_norm = self.input_norm(x)
        x_proj = F.relu(self.input_projection(x_norm)) + x_norm  # Residual connection
        
        # Get gating weights
        gating_weights = self.gating_network(x_proj)
        
        # Add noise during training for load balancing
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gating_weights) * self.noise_std
            gating_weights = gating_weights + noise
            gating_weights = F.softmax(gating_weights, dim=-1)
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_proj))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
        
        # Apply sparse MoE if top_k < num_experts
        if self.top_k < self.num_experts:
            # Select top-k experts
            top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_weights, dim=-1)
            
            # Gather outputs from top-k experts
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k)
            selected_outputs = expert_outputs[batch_indices, top_k_indices]  # (batch, top_k, output_dim)
            
            # Weighted combination
            output = torch.sum(selected_outputs * top_k_weights.unsqueeze(-1), dim=1)
        else:
            # Use all experts
            output = torch.sum(expert_outputs * gating_weights.unsqueeze(-1), dim=1)
        
        # Post-process output
        output = self.output_norm(output)
        
        # Return output and gating weights for analysis
        return output, gating_weights


class AdaptiveMoERegressor(nn.Module):
    """
    Adaptive Mixture of Experts with hierarchical gating and dynamic expert selection.
    
    This variant includes:
    - Hierarchical gating (coarse-to-fine expert selection)
    - Adaptive expert capacity based on input complexity
    - Multi-scale feature extraction before expert routing
    """
    def __init__(self, input_dim, output_dim, num_experts=8, 
                 expert_hidden_dims=[128, 64], activation='relu', dropout=0.1):
        super().__init__()
        
        self.num_experts = num_experts
        
        # Multi-scale feature extraction
        self.conv1d = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_projection = nn.Linear(input_dim + 16, input_dim)
        
        # Hierarchical gating
        self.coarse_gate = GatingNetwork(input_dim, 4, 32, activation, dropout)
        self.fine_gates = nn.ModuleList([
            GatingNetwork(input_dim, 2, 32, activation, dropout)
            for _ in range(4)
        ])
        
        # Experts organized in hierarchy
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                Expert(input_dim, output_dim, expert_hidden_dims, activation, dropout)
                for _ in range(2)
            ])
            for _ in range(4)
        ])
        
        # Input/output processing
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale feature extraction
        x_norm = self.input_norm(x)
        x_conv = self.conv1d(x_norm.unsqueeze(1))  # Add channel dim: (batch, 16, input_dim)
        x_conv_pooled = self.global_pool(x_conv).squeeze(-1)  # (batch, 16)
        x_combined = torch.cat([x_norm, x_conv_pooled], dim=-1)  # (batch, input_dim + 16)
        x_proj = self.feature_projection(x_combined)
        
        # Hierarchical gating
        coarse_weights = self.coarse_gate(x_proj)  # (batch, 4)
        
        # Collect outputs from all expert groups
        group_outputs = []
        for group_idx, (fine_gate, expert_group) in enumerate(zip(self.fine_gates, self.expert_groups)):
            fine_weights = fine_gate(x_proj)  # (batch, 2)
            
            # Get outputs from experts in this group
            expert_outputs = []
            for expert in expert_group:
                expert_outputs.append(expert(x_proj))
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, 2, output_dim)
            
            # Weighted combination within group
            group_output = torch.sum(expert_outputs * fine_weights.unsqueeze(-1), dim=1)
            group_outputs.append(group_output)
        
        # Stack group outputs and apply coarse weighting
        group_outputs = torch.stack(group_outputs, dim=1)  # (batch, 4, output_dim)
        final_output = torch.sum(group_outputs * coarse_weights.unsqueeze(-1), dim=1)
        
        # Post-process
        final_output = self.output_norm(final_output)
        
        return final_output, coarse_weights
