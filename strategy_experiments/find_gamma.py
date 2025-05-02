import torch
import numpy as np
from torch.utils.data import DataLoader
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset

"""
Gamma Calibration Script

- Computes recommended gamma scaling factors based on model predictions on the training set
- Supports different position scaling methods (mean-variance scaling, tail-risk adjusted confidence)
- Prints 95th percentile raw position and corresponding gamma for risk adjustment
- Loads a pre-trained model and runs predictions on training data
"""


def compute_gamma_from_trainset(data_path, model_path, model_tag, lookback=12):

    print(f"\n Calculating gamma based on training set for {model_tag}")

    import pandas as pd
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    test_len = int(len(df) * 0.2)
    train_df = df[:-test_len]

    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)

    if "lstm" in model_tag.lower():
        model = LSTMStudentT(input_dim=ds_train.X.shape[2])
    elif "tf" in model_tag.lower():
        model = CommodityTransformer(input_dim=ds_train.X.shape[2])
    else:
        raise ValueError(f"Unknown model type for tag: {model_tag}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_mu_train, all_sigma_train, all_nu_train = [], [], []

    with torch.no_grad():
        for batch in train_loader:
            x_train, _ = batch
            mu_train, sigma_train, nu_train = model(x_train)
            all_mu_train.append(mu_train)
            all_sigma_train.append(sigma_train)
            all_nu_train.append(nu_train)

    mu_train = torch.cat(all_mu_train, dim=0).detach().cpu().numpy()
    sigma_train = torch.cat(all_sigma_train, dim=0).detach().cpu().numpy()
    nu_train = torch.cat(all_nu_train, dim=0).detach().cpu().numpy()

    def compute_gamma(mu, sigma, nu, scaling_method='mean_variance_scaling', target_max_position=1.0):
        if scaling_method == 'mean_variance_scaling':
            variance = (sigma ** 2) * (nu / (nu - 2))
            raw_position = mu / variance
        elif scaling_method == 'tail_risk_adjusted_confidence':
            effective_nu = np.maximum(nu, 2.1)
            raw_position = mu / (sigma * (effective_nu / (effective_nu - 2)))
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        abs_position = np.abs(raw_position)
        pos_95 = np.percentile(abs_position, 95)
        gamma = target_max_position / pos_95
        return pos_95, gamma

    for method in ['mean_variance_scaling', 'tail_risk_adjusted_confidence']:
        pos_95, gamma = compute_gamma(mu_train, sigma_train, nu_train, scaling_method=method)
        print(f"[{method}] 95% quantile of raw position = {pos_95:.4f}")
        print(f"[{method}] Suggested gamma = {gamma:.4f}\n")

if __name__ == "__main__":
    compute_gamma_from_trainset(
        data_path="data/gold_feat.csv",
        model_path="model_experiments/results/gold/transformer_fixedpe.pt",
        model_tag="gold_tf",
        lookback=12
    )
    compute_gamma_from_trainset(
        data_path="data/wti_feat.csv",
        model_path="model_experiments/results/oil/transformer_fixedpe.pt",
        model_tag="oil_tf",
        lookback=12
    )
