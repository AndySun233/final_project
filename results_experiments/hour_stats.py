# results_experiments/hour_stats.py

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from data.build_dataset import read_yahoo_csv

def compute_hourly_statistics(df, save_dir, tag):
    """
    输入：原始行情数据（含Datetime，Close等）
    输出：每小时的 return 均值、方差、偏度、峰度、positive比例、negative比例、zero比例
    """
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex!")

    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column!")

    # 计算简单收益率 (按Close收盘价)
    df['return'] = df['Close'].pct_change()

    # 删除第一行NaN
    df = df.dropna()

    # 提取小时
    df['Hour'] = df.index.hour

    grouped = df.groupby('Hour')['return']

    results = []
    for hour, group in grouped:
        mean = group.mean()
        var = group.var()
        skewness = skew(group)
        kurt = kurtosis(group)
        pos_ratio = np.mean(group > 0)
        neg_ratio = np.mean(group < 0)
        zero_ratio = np.mean(group == 0)
        results.append([hour, mean, var, skewness, kurt, pos_ratio, neg_ratio, zero_ratio])

    stat_df = pd.DataFrame(results, columns=['Hour', 'Mean', 'Variance', 'Skew', 'Kurtosis', 'Positive', 'Negative', 'Zero'])
    stat_df = stat_df.sort_values('Hour').reset_index(drop=True)

    save_path = os.path.join(save_dir, f"{tag}_hourly_stats.csv")
    stat_df.to_csv(save_path, index=False)
    print(f"✅ Saved hourly stats to {save_path}")

    return stat_df

if __name__ == "__main__":
    tasks = [
        ("data/gold_1h_2yr.csv", "results_experiments/results/gold", "gold"),
        ("data/wti_1h_2yr.csv", "results_experiments/results/oil", "oil"),
    ]

    for data_path, save_dir, tag in tasks:
        df = read_yahoo_csv(data_path)
        compute_hourly_statistics(df, save_dir, tag)
