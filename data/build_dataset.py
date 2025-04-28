import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def generate_feature_matrix(
    price_df, cpi_df, eia_df, vix_df, dxy_df, cross_return_series, 
    output_path="feature_matrix.csv", rolling_window=30
):
    """
    构造并导出特征矩阵（含基础特征、宏观变量、交叉商品），加入调试信息。
    """
    df = price_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(f"📊 输入行情数据行数: {df.shape[0]}")

    # === 基础特征 ===
    df["return_close"] = df["Close"].pct_change() * 10000
    df["return_high"] = df["High"].pct_change() * 10000
    df["return_low"] = df["Low"].pct_change() * 10000
    df["spread"] = df["High"] - df["Low"]
    df["ema_5"] = df["Close"].ewm(span=5).mean()
    df["ema_20"] = df["Close"].ewm(span=20).mean()
    df["ema_30"] = df["Close"].ewm(span=30).mean()
    df["ema_60"] = df["Close"].ewm(span=60).mean()

    # 滚动标准化
    for col in ["High", "Low", "Close", "spread", "ema_5", "ema_20", "ema_30", "ema_60"]:
        mean = df[col].rolling(rolling_window).mean()
        std = df[col].rolling(rolling_window).std()
        df[f"z_{col.lower()}"] = (df[col] - mean) / std

    # 波动率
    df["vol_5"] = df["return_close"].rolling(5).std()
    df["vol_10"] = df["return_close"].rolling(10).std()
    df["vol_20"] = df["return_close"].rolling(20).std()

    # 时间编码
    df["sin_time"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["cos_time"] = np.cos(2 * np.pi * df.index.hour / 24)

    # === 宏观数据 ===
    macro_df = pd.DataFrame(index=df.index)

    def fix_index(x):
        return pd.to_datetime(x.index).tz_localize(None)

    cpi_df.index = fix_index(cpi_df) + pd.DateOffset(days=1)
    macro_df["cpi_lagged"] = cpi_df.reindex(macro_df.index, method="ffill")

    eia_df.index = fix_index(eia_df) + pd.Timedelta(days=1)
    macro_df["eia_inventory_lagged"] = eia_df.reindex(macro_df.index, method="ffill")

    dxy_df.index = fix_index(dxy_df)
    macro_df["dxy_index"] = dxy_df["Close"].reindex(macro_df.index, method="ffill")

    vix_df.index = fix_index(vix_df)
    macro_df["vix_index"] = vix_df["Close"].reindex(macro_df.index, method="ffill")

    cross_return_series.index = pd.to_datetime(cross_return_series.index).tz_localize(None)
    macro_df["cross_return"] = cross_return_series.reindex(macro_df.index)

    # 合并特征
    full_df = df.join(macro_df, how="left")

    # === 修复后的 target 构造 ===
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    full_df["target"] = price_df["Close"].pct_change().shift(-1) * 10000

    # === Debug 打印 ===
    print(f"📋 合并后总特征列: {full_df.shape[1]}")
    print("🧹 NaN 列缺失数量（前几列）:\n", full_df.isna().sum().sort_values(ascending=False).head(10))

    # Debug 文件（含 NaN）
    debug_path = output_path.replace(".csv", "_debug.csv")
    full_df.to_csv(debug_path)
    print(f"🪵 Debug 文件已保存：{debug_path}")

    # Drop NaN 并保存最终数据
    final_df = full_df.dropna()
    print(f"✅ dropna 后数据行数: {final_df.shape[0]}")
    final_df.to_csv(output_path)
    print(f"✅ 最终特征矩阵已保存：{output_path}")



def read_yahoo_csv(path):
    return pd.read_csv(
        path,
        skiprows=3,
        names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Datetime"],
        index_col="Datetime"
    )

if __name__ == "__main__":
    
    # === Step 1: 加载主行情数据 ===
    gold_df = read_yahoo_csv("data/gold_1h_2yr.csv")
    oil_df = read_yahoo_csv("data/wti_1h_2yr.csv")

    # === Step 2: 加载宏观数据（必须提前准备好） ===
    cpi_df = pd.read_csv("data/cpi_monthly.csv", usecols=["DATE", "CPIAUCSL"], parse_dates=["DATE"], index_col="DATE")
    cpi_df.rename(columns={"CPIAUCSL": "CPI"}, inplace=True)
    eia_df = pd.read_csv("data/eia.csv", skiprows=4, parse_dates=["Week of"], index_col="Week of")
    eia_df.rename(columns={eia_df.columns[0]: "EIA"}, inplace=True)
    dxy_df = read_yahoo_csv("data/dxy_daily.csv")
    vix_df = read_yahoo_csv("data/vix_daily.csv")
    
    # === Step 3: 加载交叉资产 return（如原油的对数收益率） ===
    gold_return = gold_df["Close"].pct_change() * 10000
    oil_return = oil_df["Close"].pct_change() * 10000

    
    generate_feature_matrix(gold_df, cpi_df, eia_df, vix_df, dxy_df, oil_return, output_path="data/gold_feat.csv")
    generate_feature_matrix(oil_df, cpi_df, eia_df, vix_df, dxy_df, gold_return, output_path="data/wti_feat.csv")


