# results_experiments/hourly_eval_runner.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import StudentT
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate_hourly(mu, sigma, nu, y_true, timestamps, y_mean, y_std):
    """
    按小时划分并评估 NLL, DA, 95% CR, RMSE
    """
    mu_std = mu.detach().cpu().numpy()
    sigma_std = sigma.detach().cpu().numpy()
    nu_std = nu.detach().cpu().numpy()
    y_true_std = y_true.detach().cpu().numpy()
    timestamps = pd.to_datetime(timestamps)

    # 反标准化
    mu = mu_std * y_std + y_mean
    sigma = sigma_std * y_std
    y_true = y_true_std * y_std + y_mean

    results = []
    for hour in range(24):
        hours = pd.to_datetime(timestamps).dt.hour  # 把Series转成小时
        mask = hours == hour
        if np.sum(mask) == 0:
            results.append([hour, np.nan, np.nan, np.nan, np.nan])
            continue

        mu_h = mu[mask]
        sigma_h = sigma[mask]
        nu_h = nu_std[mask]
        y_h = y_true[mask]

        # NLL
        dist = StudentT(df=torch.tensor(nu_h), loc=torch.tensor(mu_h), scale=torch.tensor(sigma_h))
        nll = -dist.log_prob(torch.tensor(y_h)).mean().item()

        # DA
        da = np.mean(mu_h * y_h >= 0)

        # 95% CR
        lower = mu_h - 1.96 * sigma_h
        upper = mu_h + 1.96 * sigma_h
        cr = np.mean((y_h >= lower) & (y_h <= upper))

        # RMSE
        rmse = mean_squared_error(y_h, mu_h, squared=False)

        results.append([hour, nll, da, cr, rmse])

    df = pd.DataFrame(results, columns=["Hour", "NLL", "DA", "95% CR", "RMSE"])
    return df

def plot_hourly_metrics(df, save_dir, tag):
    """
    绘制一张图：
    4条标准化曲线（NLL, DA, 95% CR, RMSE）+
    正常时段填充背景色 +
    交叠时段蓝绿/绿橙0.25小时交替块
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)

    # 标准化指标
    df_norm = df.copy()
    for col in ["NLL", "DA", "95% CR", "RMSE"]:
        df_norm[col] = df[col] / df[col].max()

    fig, ax = plt.subplots(figsize=(14, 6))

    # 画4条标准化曲线
    ax.plot(df_norm["Hour"], df_norm["NLL"], marker='o', label="NLL")
    ax.plot(df_norm["Hour"], df_norm["DA"], marker='s', label="DA")
    ax.plot(df_norm["Hour"], df_norm["95% CR"], marker='^', label="95% CR")
    ax.plot(df_norm["Hour"], df_norm["RMSE"], marker='x', label="RMSE")

    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title(f"{tag}: Intraday Hourly Evaluation (Normalized)")

    ax.grid(True)
    ax.legend()

    # 交易时段设置
    normal_sessions = [
        (0, 7, 'lightskyblue', "Asia"),
        (9, 12, 'palegreen', "Europe"),
        (16, 21, 'navajowhite', "America"),
    ]

    overlap_sessions = [
        (7, 9, 'lightskyblue', 'palegreen', "Asia-Europe Overlap"),
        (12, 16, 'palegreen', 'navajowhite', "Europe-America Overlap"),
    ]

    ymin, ymax = ax.get_ylim()

    # 画正常时段背景
    for start, end, color, label in normal_sessions:
        ax.axvspan(start, end, color=color, alpha=0.3)

    # 画交叠时段交替块
    for start, end, color1, color2, label in overlap_sessions:
        step = 0.25  # 每0.25小时切换一次
        current = start
        toggle = True  # 交替颜色
        while current < end:
            next_step = min(current + step, end)
            ax.axvspan(current, next_step,
                       color=color1 if toggle else color2,
                       alpha=0.3)
            toggle = not toggle
            current = next_step

    # 标注市场名字
    session_labels = [
        (3.5, 'Asia', 'blue'),
        (10.5, 'Europe', 'green'),
        (18.5, 'America', 'orange'),
    ]
    for x_pos, label, color in session_labels:
        ax.text(x_pos, ymax + 0.02, label,
                ha='center', va='bottom',
                color=color, fontsize=12)

    ax.set_xlim(0, 23)
    ax.set_ylim(ymin, ymax + 0.1)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{tag.lower()}_hourly_eval_combined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved combined hourly plot to {save_path}")




def run_hourly_analysis(data_path, model_path, lookback=6, tag="gold"):
    print(f"\n🚀 Running hourly analysis for {tag}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    test_len = int(len(df) * 0.2)
    train_df = df[:-test_len]
    test_df = df[-test_len:]

    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    ds_test = TimeSeriesDataset(test_df, lookback=lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False)

    # 选择模型
    if "lstm" in tag.lower():
        model = LSTMStudentT(input_dim=ds_test.X.shape[2])
    elif "tf" in tag.lower():
        model = CommodityTransformer(input_dim=ds_test.X.shape[2])
    else:
        raise ValueError(f"Unknown model type for tag: {tag}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 收集预测
    all_mu, all_sigma, all_nu, all_y, all_times = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            mu, sigma, nu = model(x)
            batch_start = batch_idx * 128
            batch_end = batch_start + len(y)
            timestamps = pd.Series(test_df.index[batch_start:batch_end])
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_nu.append(nu)
            all_y.append(y)
            all_times.append(timestamps)

    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    nu = torch.cat(all_nu, dim=0)
    y_true = torch.cat(all_y, dim=0)
    timestamps = pd.concat(all_times)

    # 按小时评估
    df_hourly = evaluate_hourly(mu, sigma, nu, y_true, timestamps, 
                                y_mean=ds_train.y_mean.item(), 
                                y_std=ds_train.y_std.item())

    print("\n📊 Hourly Evaluation Table:")
    print(df_hourly.round(4))

    # 保存结果
    save_dir = os.path.join("results_experiments", "results", tag)
    os.makedirs(save_dir, exist_ok=True)

    df_hourly.to_csv(os.path.join(save_dir, "hourly_eval_metrics.csv"), index=False)
    plot_hourly_metrics(df_hourly, save_dir, tag)

    print(f"✅ Saved hourly results to {save_dir}")

if __name__ == "__main__":
    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold/transformer_fixedpe.pt", "gold_tf"),
        ("data/gold_feat.csv", "model_experiments/results/gold/final_lstm.pt", "gold_lstm"),
        ("data/wti_feat.csv", "model_experiments/results/oil/transformer_fixedpe.pt", "oil_tf"),
        ("data/wti_feat.csv", "model_experiments/results/oil/final_lstm.pt", "oil_lstm"),
    ]
    for data_path, model_path, tag in tasks:
        run_hourly_analysis(data_path, model_path, lookback=12, tag=tag)
