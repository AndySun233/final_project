# generate_prediction_csv.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset

def read_yahoo_csv(path):
    return pd.read_csv(
        path,
        skiprows=3,
        names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Datetime"],
        index_col="Datetime"
    )

def save_model_predictions(data_path, model_path, market_data_path, save_csv_path, lookback=6, tag="gold"):
    print(f"\n Running prediction extraction for {tag}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    test_len = int(len(df) * 0.2)
    train_df = df[:-test_len]
    test_df = df[-test_len:]

    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    ds_test = TimeSeriesDataset(test_df, lookback=lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False)

    if "lstm" in tag.lower():
        model = LSTMStudentT(input_dim=ds_test.X.shape[2])
    elif "tf" in tag.lower():
        model = CommodityTransformer(input_dim=ds_test.X.shape[2])
    else:
        raise ValueError(f"Unknown model type for tag: {tag}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_mu, all_sigma, all_nu = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch 
            mu, sigma, nu = model(x)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_nu.append(nu)

    mu = torch.cat(all_mu, dim=0).detach().cpu().numpy()
    sigma = torch.cat(all_sigma, dim=0).detach().cpu().numpy()
    nu = torch.cat(all_nu, dim=0).detach().cpu().numpy()

    timestamps = test_df.index[lookback:] 
    assert len(timestamps) == len(mu), "Timestamp and prediction length mismatch!"

    df_market = read_yahoo_csv(market_data_path)

    df_market.index = df_market.index.tz_localize(None)

    df_market["ema_20"] = df_market["Close"].ewm(span=20, adjust=False).mean()

    df_preds = pd.DataFrame({
        'timestamp': timestamps,
        'mu': mu.flatten(),
        'sigma': sigma.flatten(),
        'nu': nu.flatten()
    })

    df_preds = pd.merge(df_preds, df_market, left_on="timestamp", right_index=True, how="inner")

    df_preds = df_preds[["timestamp", "mu", "sigma", "nu", "Close", "ema_20"]]

    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df_preds.to_csv(save_csv_path, index=False)

    print(f"Final Prediction CSV (with close and ema_20) saved at {save_csv_path}")

if __name__ == "__main__":
    tasks = [
        (
            "data/gold_feat.csv",
            "model_experiments/results/gold/transformer_fixedpe.pt",
            "data/gold_1h_2yr.csv",
            "strategy_experiments/results/gold_tf_prediction.csv",
            "gold_tf"
        ),
        (
            "data/gold_feat.csv",
            "model_experiments/results/gold/final_lstm.pt",
            "data/gold_1h_2yr.csv",
            "strategy_experiments/results/gold_lstm_prediction.csv",
            "gold_lstm"
        ),
        (
            "data/wti_feat.csv",
            "model_experiments/results/oil/transformer_fixedpe.pt",
            "data/wti_1h_2yr.csv",
            "strategy_experiments/results/oil_tf_prediction.csv",
            "oil_tf"
        ),
        (
            "data/wti_feat.csv",
            "model_experiments/results/oil/final_lstm.pt",
            "data/wti_1h_2yr.csv",
            "strategy_experiments/results/oil_lstm_prediction.csv",
            "oil_lstm"
        )
    ]
    for data_path, model_path, market_data_path, save_csv_path, tag in tasks:
        save_model_predictions(data_path, model_path, market_data_path, save_csv_path, lookback=12, tag=tag)
