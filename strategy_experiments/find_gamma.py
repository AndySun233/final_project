import torch
import numpy as np
from torch.utils.data import DataLoader
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset

def compute_gamma_from_trainset(data_path, model_path, model_tag, lookback=12):
    """
    从训练集直接算gamma，不保存中间文件
    """
    print(f"\n🚀 Calculating gamma based on training set for {model_tag}")

    # === 1. 加载已经做特征工程的数据 ===
    import pandas as pd
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    test_len = int(len(df) * 0.2)
    train_df = df[:-test_len]

    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)

    # === 2. 加载模型 ===
    if "lstm" in model_tag.lower():
        model = LSTMStudentT(input_dim=ds_train.X.shape[2])
    elif "tf" in model_tag.lower():
        model = CommodityTransformer(input_dim=ds_train.X.shape[2])
    else:
        raise ValueError(f"Unknown model type for tag: {model_tag}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # === 3. 跑训练集，拿 mu, sigma, nu ===
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

    # === 4. 计算gamma ===
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
        pos_95 = np.percentile(abs_position, 95)  # 取95分位
        gamma = target_max_position / pos_95
        return pos_95, gamma

    # === 5. 输出gamma
    for method in ['mean_variance_scaling', 'tail_risk_adjusted_confidence']:
        pos_95, gamma = compute_gamma(mu_train, sigma_train, nu_train, scaling_method=method)
        print(f"📈 [{method}] 95% quantile of raw position = {pos_95:.4f}")
        print(f"✅ [{method}] Suggested gamma = {gamma:.4f}\n")

# ✨ 示例调用
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
