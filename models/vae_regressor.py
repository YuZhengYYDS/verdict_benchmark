import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import get_activation

class VAERegressor(nn.Module):
    """
    Variational Autoencoder with regression head for VERDICT parameter prediction.
    Uses latent space regularization to learn meaningful representations.
    """
    def __init__(self, input_dim, output_dim, latent_dim=32, hidden_dims=[128, 64], activation='relu', dropout=0.1, beta=1.0, alpha=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.alpha = alpha
        
        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(get_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        decode_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        for i in range(len(decode_dims) - 1):
            decoder_layers.append(nn.Linear(decode_dims[i], decode_dims[i+1]))
            if i < len(decode_dims) - 2:
                decoder_layers.append(get_activation(activation))
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        pred_y = self.regressor(z)
        
        return pred_y, recon_x, mu, logvar

    def loss_function(self, pred_y, target_y, recon_x, x, mu, logvar):
        """
        Combined loss: regression + reconstruction + KL divergence
        """
        mse_loss = F.mse_loss(pred_y, target_y)
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return mse_loss + self.alpha * recon_loss + self.beta * kl_loss
