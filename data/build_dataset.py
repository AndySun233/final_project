import pandas as pd
import numpy as np

"""
Feature Matrix Generator

- Inputs: price data, macro indicators (CPI, EIA, DXY, VIX), cross-commodity returns
- Outputs: cleaned feature matrix CSV + debug CSV
- Features: price stats, volatility, rolling z-scores, macro lags, time encodings
- Purpose: prepare data for probabilistic forecasting models
"""


def generate_feature_matrix(
    price_df, cpi_df, eia_df, vix_df, dxy_df, cross_return_series, 
    output_path="feature_matrix.csv", rolling_window=30
):
    df = price_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(f" input line: {df.shape[0]}")

    df["return_close"] = df["Close"].pct_change() * 10000
    df["return_high"] = df["High"].pct_change() * 10000
    df["return_low"] = df["Low"].pct_change() * 10000
    df["spread"] = df["High"] - df["Low"]
    df["ema_5"] = df["Close"].ewm(span=5).mean()
    df["ema_20"] = df["Close"].ewm(span=20).mean()
    df["ema_30"] = df["Close"].ewm(span=30).mean()
    df["ema_60"] = df["Close"].ewm(span=60).mean()

    for col in ["High", "Low", "Close", "spread", "ema_5", "ema_20", "ema_30", "ema_60"]:
        mean = df[col].rolling(rolling_window).mean()
        std = df[col].rolling(rolling_window).std()
        df[f"z_{col.lower()}"] = (df[col] - mean) / std

    df["vol_5"] = df["return_close"].rolling(5).std()
    df["vol_10"] = df["return_close"].rolling(10).std()
    df["vol_20"] = df["return_close"].rolling(20).std()

    df["sin_time"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["cos_time"] = np.cos(2 * np.pi * df.index.hour / 24)

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

    full_df = df.join(macro_df, how="left")

    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    full_df["target"] = price_df["Close"].pct_change().shift(-1) * 10000

    # === Debug print ===
    print(f" total line: {full_df.shape[1]}")
    print(" NaN :\n", full_df.isna().sum().sort_values(ascending=False).head(10))

    # Debug
    debug_path = output_path.replace(".csv", "_debug.csv")
    full_df.to_csv(debug_path)
    print(f"debug：{debug_path}")

    # Drop NaN and save
    final_df = full_df.dropna()
    print(f" line after dropna : {final_df.shape[0]}")
    final_df.to_csv(output_path)
    print(f"feat save to：{output_path}")



def read_yahoo_csv(path):
    return pd.read_csv(
        path,
        skiprows=3,
        names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Datetime"],
        index_col="Datetime"
    )

if __name__ == "__main__":
    
    # === Step 1===
    gold_df = read_yahoo_csv("data/gold_1h_2yr.csv")
    oil_df = read_yahoo_csv("data/wti_1h_2yr.csv")

    # === Step 2 ===
    cpi_df = pd.read_csv("data/cpi_monthly.csv", usecols=["DATE", "CPIAUCSL"], parse_dates=["DATE"], index_col="DATE")
    cpi_df.rename(columns={"CPIAUCSL": "CPI"}, inplace=True)
    eia_df = pd.read_csv("data/eia.csv", skiprows=4, parse_dates=["Week of"], index_col="Week of")
    eia_df.rename(columns={eia_df.columns[0]: "EIA"}, inplace=True)
    dxy_df = read_yahoo_csv("data/dxy_daily.csv")
    vix_df = read_yahoo_csv("data/vix_daily.csv")
    
    # === Step 3 ===
    gold_return = gold_df["Close"].pct_change() * 10000
    oil_return = oil_df["Close"].pct_change() * 10000

    
    generate_feature_matrix(gold_df, cpi_df, eia_df, vix_df, dxy_df, oil_return, output_path="data/gold_feat.csv")
    generate_feature_matrix(oil_df, cpi_df, eia_df, vix_df, dxy_df, gold_return, output_path="data/wti_feat.csv")


