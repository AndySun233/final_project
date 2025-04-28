import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from train.train_lstm import train_model as train_lstm
from train.train_tf import train_model as train_tf
from models.lstm_model import LSTMStudentT
from models.tf_model import CommodityTransformer
from utils.dataset import TimeSeriesDataset
from losses.loss import nll_only
from eval.evaluate import evaluate_model
from utils.seeds import set_seed

def run_experiment(data_path, model_class, train_fn, save_dir, model_tag, results_list):
    print(f"\n🚀 Running: {save_dir}/{model_tag}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    ds_train = TimeSeriesDataset(train_df, lookback=3)
    ds_test = TimeSeriesDataset(test_df, lookback=3,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=128)

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_tag.lower()}.pt")

    input_dim = df.shape[1] - 1
    model = model_class(input_dim=input_dim)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"✅ Loaded model from {model_save_path}")
    else:
        model = train_fn(
            train_loader=train_loader,
            input_dim=input_dim,
            epochs=70,
            lr=1e-4,
            model_path=model_save_path,
            loss_fn=nll_only,
            model=model
        )

    result_text = evaluate_model(
        model, train_loader, test_loader,
        ds_train.y_mean, ds_train.y_std,
        train_df=train_df, test_df=test_df,
        prefix=model_tag,
        save_dir=save_dir,
        save_txt=False
    )
    results_list.append(result_text)


if __name__ == "__main__":
    set_seed(42)
    # Gold
    gold_results = []
    run_experiment(
        "data/gold_feat.csv", 
        LSTMStudentT, train_lstm, 
        save_dir="model_experiments/results/gold", 
        model_tag="LSTM",
        results_list=gold_results
    )
    run_experiment(
        "data/gold_feat.csv", 
        CommodityTransformer, train_tf, 
        save_dir="model_experiments/results/gold", 
        model_tag="Transformer",
        results_list=gold_results
    )
    with open("model_experiments/results/gold/model_compare.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(gold_results))

    # Oil
    oil_results = []
    run_experiment(
        "data/wti_feat.csv", 
        LSTMStudentT, train_lstm, 
        save_dir="model_experiments/results/oil", 
        model_tag="LSTM",
        results_list=oil_results
    )
    run_experiment(
        "data/wti_feat.csv", 
        CommodityTransformer, train_tf, 
        save_dir="model_experiments/results/oil", 
        model_tag="Transformer",
        results_list=oil_results
    )
    with open("model_experiments/results/oil/model_compare.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(oil_results))
