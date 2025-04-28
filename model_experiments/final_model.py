import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.tf_model import CommodityTransformer
from models.lstm_model import LSTMStudentT
from utils.dataset import TimeSeriesDataset
from losses.loss import composite_loss_v2
from eval.evaluate import evaluate_plaintext
from train.train_lstm import train_model as train_lstm
from train.train_tf import train_model as train_tf
from utils.seeds import set_seed

def run_final_experiment(data_path, save_dir):
    print(f"\n🚀 Running final models for {save_dir}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    lookback = 12
    ds_train = TimeSeriesDataset(train_df, lookback=lookback)
    ds_test = TimeSeriesDataset(test_df, lookback=lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)
    print(f"训练数据量（train_df样本数）: {len(ds_train)}")

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=128)

    os.makedirs(save_dir, exist_ok=True)

    results_list = []

    # === 训练 LSTM 模型 ===
    lstm_model = LSTMStudentT(input_dim=df.shape[1] - 1)
    lstm_model_path = os.path.join(save_dir, "final_lstm.pt")

    if os.path.exists(lstm_model_path):
        lstm_model.load_state_dict(torch.load(lstm_model_path))
        print(f"✅ Loaded existing final_lstm.pt")
    else:
        lstm_model = train_lstm(
            train_loader=train_loader,
            input_dim=ds_train.X.shape[2],
            epochs=70,
            lr=1e-4,
            model_path=lstm_model_path,
            loss_fn=composite_loss_v2,
            model=lstm_model
        )
        lstm_model.load_state_dict(torch.load(lstm_model_path))
        print(f"✅ Trained and saved final_lstm.pt")

    # 测试集评估 LSTM
    lstm_result = evaluate_plaintext(
        model=lstm_model,
        dataloader_test=test_loader,
        y_mean=ds_train.y_mean,
        y_std=ds_train.y_std,
        loss_fn=composite_loss_v2,
        tag="LSTM",
        save_path=None
    )
    results_list.append(lstm_result)

    result_file = os.path.join(save_dir, "final_result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results_list))

    print(f"✅ Final results saved to {result_file}")

if __name__ == "__main__":
    set_seed(42)

    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold"),
        ("data/wti_feat.csv", "model_experiments/results/oil"),
    ]

    for data_path, save_dir in tasks:
        run_final_experiment(data_path, save_dir)
