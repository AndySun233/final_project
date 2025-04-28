import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.tf_model import CommodityTransformer
from utils.dataset import TimeSeriesDataset
from losses.loss import composite_loss_v2
from eval.evaluate import evaluate_plaintext
from train.train_tf import train_model as train_tf
from utils.seeds import set_seed

def prepare_dataloader(csv_path, lookback):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    ds_train = TimeSeriesDataset(train_df, lookback)
    ds_test = TimeSeriesDataset(test_df, lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)
    return ds_train, ds_test, ds_train.y_mean, ds_train.y_std

def run_single_experiment(csv_path, save_dir, lookback, model_class, train_fn, results_list):
    tag = f"{model_class.__name__} | Lookback={lookback}"
    print(f"\n🔍 Running: {save_dir}/{tag}")

    ds_train, ds_test, y_mean, y_std = prepare_dataloader(csv_path, lookback)
    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=128)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"temp_model.pt")
    if os.path.exists(model_path):
        os.remove(model_path)

    model_instance = model_class(input_dim=ds_train.X.shape[2])
    model = train_fn(
        train_loader=train_loader,
        input_dim=ds_train.X.shape[2],
        epochs=70,
        lr=1e-4,
        model_path=model_path,
        loss_fn=composite_loss_v2,
        model=model_instance
    )

    # 加载最优模型权重
    model.load_state_dict(torch.load(model_path))

    # 测试集评估
    result = evaluate_plaintext(
        model=model,
        dataloader_test=test_loader,
        y_mean=y_mean,
        y_std=y_std,
        loss_fn=composite_loss_v2,
        tag=tag,
        save_path=None
    )

    results_list.append(result)

if __name__ == "__main__":
    set_seed(42)
    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold"),
        ("data/wti_feat.csv", "model_experiments/results/oil"),
    ]
    lookbacks = [3, 6, 12, 24]

    for csv_path, save_dir in tasks:
        os.makedirs(save_dir, exist_ok=True)
        results_list = []
        for lookback in lookbacks:
            run_single_experiment(
                csv_path=csv_path,
                save_dir=save_dir,
                lookback=lookback,
                model_class=CommodityTransformer,  # 统一用 Transformer
                train_fn=train_tf,
                results_list=results_list
            )

        # 保存结果
        output_file = os.path.join(save_dir, "lookback_compare.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results_list))
        print(f"✅ 写入完成：{output_file}")
