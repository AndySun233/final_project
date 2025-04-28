import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMStudentT(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.log_sigma_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.log_nu_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.log_nu_head[-1].bias, 0.5413)
        for layer in list(self.shared) + list(self.mu_head) + list(self.log_sigma_head):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        feat = self.shared(last)
        mu = self.mu_head(feat).squeeze(-1)
        sigma = F.softplus(self.log_sigma_head(feat).squeeze(-1)) + 1e-3
        nu = F.softplus(self.log_nu_head(feat).squeeze(-1)) + 2.0
        return mu, sigma, nu
