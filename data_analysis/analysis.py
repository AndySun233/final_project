import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, t, kstest

def evaluate_distribution_fit(series, name="Asset"):
    # 清洗
    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\n===== [{name}] 分布拟合评价 =====")

    # 参数拟合
    mu, sigma = norm.fit(series)
    df_t, loc_t, scale_t = t.fit(series)

    # === KS检验 ===
    ks_norm = kstest(series, 'norm', args=(mu, sigma))
    ks_t = kstest(series, 't', args=(df_t, loc_t, scale_t))

    print(f"[KS检验] 正态分布: D = {ks_norm.statistic:.4f}, p = {ks_norm.pvalue:.4f}")
    print(f"[KS检验] 学生t分布: D = {ks_t.statistic:.4f}, p = {ks_t.pvalue:.4f}")

    # === AIC计算 ===
    loglik_norm = np.sum(norm.logpdf(series, loc=mu, scale=sigma))
    loglik_t = np.sum(t.logpdf(series, df=df_t, loc=loc_t, scale=scale_t))

    # 参数个数: 正态 = 2，t分布 = 3
    aic_norm = 2*2 - 2*loglik_norm
    aic_t = 2*3 - 2*loglik_t

    print(f"[AIC] 正态分布: AIC = {aic_norm:.2f}")
    print(f"[AIC] 学生t分布: AIC = {aic_t:.2f}")

    if aic_t < aic_norm:
        print("✅ 学生t分布拟合更优")
    else:
        print("✅ 正态分布拟合更优")

# ===== 读取文件 =====
gold_file = "data/gold_1h_2yr.csv"
wti_file = "data/wti_1h_2yr.csv"
cols = ["Datetime", "Close", "High", "Low", "Open", "Volume"]

gold = pd.read_csv(gold_file, skiprows=3, names=cols, parse_dates=["Datetime"], index_col="Datetime")
wti = pd.read_csv(wti_file, skiprows=3, names=cols, parse_dates=["Datetime"], index_col="Datetime")

# ===== 计算线性收益率（线性 return）=====
gold["linear_return"] = gold["Close"].pct_change()
wti["linear_return"] = wti["Close"].pct_change()
gold.dropna(inplace=True)
wti.dropna(inplace=True)

# ===== 描述统计函数 =====
def describe(series):
    return {
        "Mean": np.mean(series),
        "Std Dev": np.std(series),
        "Skewness": skew(series),
        "Kurtosis": kurtosis(series)
    }

# ===== 打印统计值 =====
print("===== Gold Stats =====")
print(describe(gold["linear_return"]))
print("\n===== WTI Stats =====")
print(describe(wti["linear_return"]))

# ===== 分布拟合评估 =====
evaluate_distribution_fit(gold["linear_return"], name="Gold")
evaluate_distribution_fit(wti["linear_return"], name="WTI")

# ===== 绘图 + 保存 =====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(gold["linear_return"], bins=100, density=True, alpha=0.7)
plt.title("Gold Linear Return Distribution")

plt.subplot(1, 2, 2)
plt.hist(wti["linear_return"], bins=100, density=True, alpha=0.7)
plt.title("WTI Linear Return Distribution")

plt.tight_layout()
plt.savefig("data_analysis/linear_return_distributions.png", dpi=300)
plt.show()
