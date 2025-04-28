# results_experiments/threshold_eval_runner.py

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import StudentT
from sklearn.metrics import mean_squared_error
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset

def compute_threshold_metrics(mu, sigma, nu, y_true, y_mean, y_std, thresholds=None):
    """
    计算不同阈值下的指标：Proportion, NLL, DA, 95% CR, RMSE
    ✅ 加了反标准化，统一和训练时标准
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 2.5, 0.25)

    results = []

    # 标准化域
    mu_std = mu.detach().cpu().numpy()
    sigma_std = sigma.detach().cpu().numpy()
    nu_std = nu.detach().cpu().numpy()
    y_true_std = y_true.detach().cpu().numpy()

    # 反标准化域
    mu = mu_std * y_std + y_mean
    sigma = sigma_std * y_std
    y_true = y_true_std * y_std + y_mean

    signal_strength = np.abs(y_true_std) / sigma_std  # ❗️ 注意，这里用标准化的signal strength筛选！

    for t in thresholds:
        mask = signal_strength >= t  # ✅ 改成 >=，保证threshold=0不漏样本
        proportion = np.mean(mask)

        if np.sum(mask) == 0:
            results.append([t, proportion, np.nan, np.nan, np.nan, np.nan])
            continue

        mu_t = mu[mask]
        sigma_t = sigma[mask]
        nu_t = nu_std[mask]
        y_t = y_true[mask]

        # NLL
        dist = StudentT(df=torch.tensor(nu_t), loc=torch.tensor(mu_t), scale=torch.tensor(sigma_t))
        nll = -dist.log_prob(torch.tensor(y_t)).mean().item()

        # Directional Accuracy
        da = np.mean(mu_t * y_t >= 0)

        # 95% Coverage Rate
        lower = mu_t - 2 * sigma_t
        upper = mu_t + 2 * sigma_t
        cr = np.mean((y_t >= lower) & (y_t <= upper))

        # RMSE
        rmse = mean_squared_error(y_t, mu_t, squared=False)

        results.append([t, proportion, nll, da, cr, rmse])

    df = pd.DataFrame(results, columns=["Threshold", "Proportion", "NLL", "DA", "95% CR", "RMSE"])
    return df

def plot_threshold_metrics(df, save_path):
    """
    绘制标准化后的变化率曲线
    每个指标都除以自己Threshold=0时的初始值
    """
    import matplotlib.pyplot as plt

    # === 标准化 ===
    df_norm = df.copy()
    for col in ["NLL", "DA", "95% CR", "RMSE"]:
        df_norm[col] = df[col] / df[col].iloc[0]  # 除以自己在threshold=0时候的数值

    # === 开始画图 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Normalized Metric Value (relative to Threshold=0)")

    ax.plot(df_norm["Threshold"], df_norm["NLL"], label="NLL", marker='o')
    ax.plot(df_norm["Threshold"], df_norm["DA"], label="DA", marker='s')
    ax.plot(df_norm["Threshold"], df_norm["95% CR"], label="95% CR", marker='^')
    ax.plot(df_norm["Threshold"], df_norm["RMSE"], label="RMSE", marker='x')

    ax.legend()
    ax.set_title("Threshold-based Evaluation Metrics (Normalized)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_threshold_analysis(data_path, model_path, lookback=6, tag="gold"):
    print(f"\n🚀 Running threshold analysis for {tag}")

    # === 1. 加载数据 ===
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    test_len = int(len(df) * 0.2)
    train_df = df[:-test_len]
    test_df = df[-test_len:]

    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    ds_test = TimeSeriesDataset(test_df, lookback=lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False)

    # === 2. 加载模型 ===
    if "lstm" in tag.lower():
        model = LSTMStudentT(input_dim=ds_test.X.shape[2])
    elif "tf" in tag.lower():
        model = CommodityTransformer(input_dim=ds_test.X.shape[2])
    else:
        raise ValueError(f"Unknown model type for tag: {tag}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # === 3. 收集预测
    all_mu, all_sigma, all_nu, all_y = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            mu, sigma, nu = model(x)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_nu.append(nu)
            all_y.append(y)

    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    nu = torch.cat(all_nu, dim=0)
    y_true = torch.cat(all_y, dim=0)

    # === 4. 阈值分析
    df_metrics = compute_threshold_metrics(mu, sigma, nu, y_true, 
                                            y_mean=ds_train.y_mean.item(), 
                                            y_std=ds_train.y_std.item())

    print("\n📊 Threshold-based Evaluation Table:")
    print(df_metrics.round(4))

    # === 5. 保存结果
    save_dir = os.path.join("results_experiments", "results", tag)
    os.makedirs(save_dir, exist_ok=True)

    df_metrics.to_csv(os.path.join(save_dir, "threshold_eval_metrics.csv"), index=False)
    plot_threshold_metrics(df_metrics, save_path=os.path.join(save_dir, "threshold_eval_plot.png"))
    print(f"✅ Results saved to {save_dir}")

if __name__ == "__main__":
    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold/transformer_fixedpe.pt", "gold_tf"),
        ("data/wti_feat.csv", "model_experiments/results/oil/transformer_fixedpe.pt", "oil_tf"),
    ]
    for data_path, model_path, tag in tasks:
        run_threshold_analysis(data_path, model_path, lookback=12, tag=tag)
