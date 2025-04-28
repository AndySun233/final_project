import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]


class CommodityTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout))
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.log_sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.log_nu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        nn.init.constant_(self.log_nu_head[-1].bias, 0.5413)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        B, T, _ = x.shape
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        last = x[:, -1, :]
        feat = self.shared(last)
        mu = self.mu_head(feat).squeeze(-1)
        sigma = F.softplus(self.log_sigma_head(feat).squeeze(-1)) + 1e-3
        nu = F.softplus(self.log_nu_head(feat).squeeze(-1)) + 2.0
        return mu, sigma, nu