from train.train_tf import train_model as train_tf
from models.tf_learnable import CommodityTransformer as TransformerLearnable
from models.tf_model import CommodityTransformer as TransformerFixed
from utils.dataset import TimeSeriesDataset
from losses.loss import composite_loss_v2, nll_only
from eval.evaluate import evaluate_plaintext
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd, torch, os
from utils.seeds import set_seed

"""
Positional Encoding Ablation Experiment

- Compares Transformer models with learnable vs. fixed positional encoding
- Trains on gold and oil datasets with a fixed lookback window
- Uses composite loss for probabilistic prediction
- Evaluates models on a test set and logs plain-text metrics (Loss, DA, CI_95, RMSE)
"""

def run_experiment(data_path, model_class, train_fn, save_dir, model_tag, results_list):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

    lookback = 12
    ds_train = TimeSeriesDataset(train_df, lookback)
    ds_test = TimeSeriesDataset(test_df, lookback,
                                 x_mean=ds_train.x_mean, x_std=ds_train.x_std,
                                 y_mean=ds_train.y_mean, y_std=ds_train.y_std)

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=128)

    os.makedirs(save_dir, exist_ok=True)

    model = model_class(input_dim=ds_train.X.shape[2])
    save_model_name = model_tag.lower().replace("-", "_") + ".pt"
    model_path = os.path.join(save_dir, save_model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    model = train_fn(
        train_loader=train_loader,
        input_dim=ds_train.X.shape[2],
        epochs=70,
        lr=1e-4,
        model_path=model_path,
        loss_fn=composite_loss_v2,
        model=model
    )

    model.load_state_dict(torch.load(model_path))

    result = evaluate_plaintext(
        model=model,
        dataloader_test=test_loader,
        y_mean=ds_train.y_mean,
        y_std=ds_train.y_std,
        loss_fn=composite_loss_v2,
        tag=model_tag,
        save_path=None
    )
    results_list.append(result)

if __name__ == "__main__":
    set_seed(42)
    tasks = [
        ("data/gold_feat.csv", "model_experiments/results/gold"),
        ("data/wti_feat.csv", "model_experiments/results/oil"),
    ]
    models = [
        (TransformerLearnable, train_tf, "Transformer-LearnablePE"),
        (TransformerFixed, train_tf, "Transformer-FixedPE"),
    ]

    for data_path, save_dir in tasks:
        os.makedirs(save_dir, exist_ok=True)
        results_list = []
        for model_class, train_fn, model_tag in models:
            run_experiment(
                data_path=data_path,
                model_class=model_class,
                train_fn=train_fn,
                save_dir=save_dir,
                model_tag=model_tag,
                results_list=results_list
            )

        result_path = os.path.join(save_dir, "positional_encoding_ablation.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results_list))
        print(f" save to:{result_path}")
