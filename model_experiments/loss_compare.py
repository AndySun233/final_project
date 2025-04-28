import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.lstm_model import LSTMStudentT
from models.tf_model import CommodityTransformer
from utils.dataset import TimeSeriesDataset
from losses.loss import (
    nll_only,
    fixed_nu,
    composite_loss_v2,
    crps_loss,
    winkler_loss,
)
from eval.evaluate import evaluate_plaintext
from train.train_lstm import train_model as train_lstm
from train.train_tf import train_model as train_tf
from utils.seeds import set_seed

def run_ablation(data_path, save_dir, model_class, train_fn, loss_fn, model_tag, loss_tag, results_list):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    ds_train = TimeSeriesDataset(train_df, lookback=3)
    ds_test = TimeSeriesDataset(test_df, lookback=3,
                                x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=128)

    input_dim = df.shape[1] - 1
    model = model_class(input_dim=input_dim)
    model_path = os.path.join(save_dir, "temp_model.pt")

    if os.path.exists(model_path):
        os.remove(model_path)

    model = train_fn(
        train_loader=train_loader,
        input_dim=input_dim,
        epochs=70,
        lr=1e-4,
        model_path=model_path,
        loss_fn=loss_fn,
        model=model
    )

    tag = f"{model_tag} + {loss_tag}"
    result_text = evaluate_plaintext(
        model=model,
        dataloader_test=test_loader,
        y_mean=ds_train.y_mean,
        y_std=ds_train.y_std,
        loss_fn=loss_fn,
        tag=tag,
        save_path=None
    )
    results_list.append(result_text)

if __name__ == "__main__":
    set_seed(42)
    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold", CommodityTransformer, train_tf, "Transformer"),
        ("data/wti_feat.csv", "model_experiments/results/oil", CommodityTransformer, train_tf, "Transformer"),
    ]

    ablations = [
        (nll_only, "NLL"),
        (fixed_nu, "FixedNu"),
        (crps_loss, "CRPSOnly"),
        (composite_loss_v2, "CompLoss"),
    ]

    for data_path, save_dir, model_class, train_fn, model_tag in tasks:
        os.makedirs(save_dir, exist_ok=True)
        results_list = []
        for loss_fn, loss_tag in ablations:
            print(f"\n🚀 Running: {save_dir}/{model_tag} + {loss_tag}")
            run_ablation(
                data_path=data_path,
                save_dir=save_dir,
                model_class=model_class,
                train_fn=train_fn,
                loss_fn=loss_fn,
                model_tag=model_tag,
                loss_tag=loss_tag,
                results_list=results_list
            )

        # 保存
        output_file = os.path.join(save_dir, "loss_function_ablation.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results_list))
        print(f"✅ 写入完成：{output_file}")
