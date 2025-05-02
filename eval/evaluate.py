import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from torch import tensor
from losses.loss import nll_only
from contextlib import redirect_stdout
from torch.distributions import StudentT

"""
Model Evaluation & Visualization Utilities

- Evaluate probabilistic deep learning models on train/test sets
- Compute metrics: NLL loss, directional accuracy, 95% confidence interval coverage, RMSE
- Generate evaluation logs and prediction plots with confidence intervals
- Outputs saved as text files and PNGs under a specified directory
"""


def evaluate_model(model, train_loader, test_loader,
                   y_mean, y_std,
                   train_df=None, test_df=None,
                   prefix="Model", save_dir="results", save_txt=True):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{prefix.lower()}_eval.txt")

    def collect_predictions(loader):
        all_mu, all_sigma, all_nu, all_y = [], [], [], []
        with torch.no_grad():
            for X, y in loader:
                mu, sigma, nu = model(X)
                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_nu.append(nu.cpu().numpy())
                all_y.append(y.cpu().numpy())
        return (np.concatenate(all_mu),
                np.concatenate(all_sigma),
                np.concatenate(all_nu),
                np.concatenate(all_y))

    def do_evaluation():
        mu_tr_raw, sig_tr_raw, nu_tr, y_tr_raw = collect_predictions(train_loader)
        mu_te_raw, sig_te_raw, nu_te, y_te_raw = collect_predictions(test_loader)

        composite_loss_tr = nll_only(
            tensor(mu_tr_raw), tensor(sig_tr_raw), tensor(y_tr_raw), tensor(nu_tr)
        ).item()
        composite_loss_te = nll_only(
            tensor(mu_te_raw), tensor(sig_te_raw), tensor(y_te_raw), tensor(nu_te)
        ).item()

        mu_tr = mu_tr_raw * y_std + y_mean
        sig_tr = sig_tr_raw * y_std
        y_tr = y_tr_raw * y_std + y_mean

        mu_te = mu_te_raw * y_std + y_mean
        sig_te = sig_te_raw * y_std
        y_te = y_te_raw * y_std + y_mean

        def format_metrics(mu, sigma, nu, y, composite_loss, split="Test"):
            da = np.mean(mu * y >= 0)
            cr95 = np.mean((y >= mu - 2 * sigma) & (y <= mu + 2 * sigma))
            rmse = np.sqrt(np.mean((mu - y) ** 2))
            return (
                f"========== {prefix} [{split}] ==========\n"
                f"Loss: {composite_loss:.6f}, DA: {da:.4f}, CI_95: {cr95:.4f}, RMSE: {rmse:.4f}\n"
            )

        text = ""
        text += format_metrics(mu_tr, sig_tr, nu_tr, y_tr, composite_loss_tr, split="Train")
        ts_tr = (train_df.index[-len(y_tr):] if train_df is not None else None)
        visualize_predictions(mu_tr, sig_tr, y_tr, timestamps=ts_tr, prefix=prefix, tag="train", save_dir=save_dir)
        text += format_metrics(mu_te, sig_te, nu_te, y_te, composite_loss_te, split="Test")
        ts_te = (test_df.index[:len(y_te)] if test_df is not None else None)
        visualize_predictions(mu_te, sig_te, y_te, timestamps=ts_te, prefix=prefix, tag="test", save_dir=save_dir)
        return text

    text = do_evaluation()
    if save_txt:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"saved to：{log_path}")
    return text


def visualize_predictions(mu, sigma, y_true,
                          timestamps=None, prefix="Model", tag="test", save_dir="results"):

    max_points = None
    if max_points:
        mu, sigma, y_true = mu[:max_points], sigma[:max_points], y_true[:max_points]
        if timestamps is not None:
            timestamps = timestamps[:max_points]

    mask = ~np.isnan(mu) & ~np.isnan(sigma) & ~np.isnan(y_true)
    mu, sigma, y_true = mu[mask], sigma[mask], y_true[mask]
    if timestamps is not None:
        timestamps = pd.to_datetime(np.array(timestamps)[mask])
    x = timestamps if timestamps is not None else np.arange(len(mu))

    plt.figure(figsize=(15, 6))
    plt.plot(x, y_true, label="Actual", color="black", alpha=0.5)
    plt.plot(x, mu, label="Predicted Mean", color="blue")
    plt.fill_between(x, mu - 2 * sigma, mu + 2 * sigma, color="blue", alpha=0.2, label="95% CI")

    plt.title(f"{prefix} [{tag.upper()}]: Prediction with 95% Confidence Interval")
    plt.xlabel("Datetime")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)

    if isinstance(x[0], pd.Timestamp):
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix.lower().replace(' ', '_')}_{tag}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def evaluate_plaintext(model, dataloader_test, y_mean, y_std, loss_fn, tag, save_path=None):
    import torch
    import numpy as np
    from torch import tensor
    import os

    model.eval()
    all_mu, all_sigma, all_nu, all_y = [], [], [], []
    with torch.no_grad():
        for X, y in dataloader_test:
            mu, sigma, nu = model(X)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_nu.append(nu.cpu().numpy())
            all_y.append(y.cpu().numpy())

    mu, sigma, nu, y = map(np.concatenate, (all_mu, all_sigma, all_nu, all_y))
    composite_loss = loss_fn(tensor(mu), tensor(sigma), tensor(y), tensor(nu)).item()

    mu = mu * y_std + y_mean
    sigma = sigma * y_std
    y = y * y_std + y_mean

    da = np.mean(mu * y >= 0)
    cr95 = np.mean((y >= mu - 2 * sigma) & (y <= mu + 2 * sigma))
    rmse = np.sqrt(np.mean((mu - y) ** 2))

    result_text = f"""
    ========== {tag} [Test] ==========
    Loss: {composite_loss:.6f}, DA: {da:.4f}, CI_95: {cr95:.4f}, RMSE: {rmse:.4f}
    """

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result_text.strip() + "\n")
        print(f"save to {save_path}")

    return result_text.strip()

