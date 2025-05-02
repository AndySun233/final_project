# backtest.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from strategy_experiments.strategy import strategy_mapper

"""
Backtesting Script

- Runs backtest simulations on prediction CSV files using multiple trading strategies
- Computes key performance metrics: total return, Sharpe ratio, max drawdown, win rate
- Saves performance summary table and account value plot
- Supports optional active trading hours filtering
"""


INIT_CAPITAL = 1_000_000  
FEE_RATE = 0.0001        
SLIPPAGE_RATE = 0.0005    
MAX_EXPOSURE = 1        

def run_backtest(pred_path, save_dir, model_tag, active_hours = None):
    print(f"\n Running backtest for {model_tag}")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(pred_path, parse_dates=['timestamp'])
    
    if active_hours is not None:
        df['hour'] = df['timestamp'].dt.hour 
        df = df[df['hour'].isin(active_hours)].reset_index(drop=True)
        print(f"Trading restricted to hours: {active_hours}")

    strategy_grid = [
        ('buy_and_hold',              {}),
        ('moving_average_crossover',  {}),
        ('threshold_only',            {}),
        ('mean_variance_scaling_0.5',  {'gamma': 0.5}),
        ('mean_variance_scaling_1',  {'gamma': 1}),
        ('mean_variance_scaling_1.5',   {'gamma': 1.5}),
        ('tail_risk_adjusted_confidence_1', {'gamma': 1}),
        ('tail_risk_adjusted_confidence_2', {'gamma': 2}),
        ('tail_risk_adjusted_confidence_3',  {'gamma': 3}),
    ]

    STRATEGY_ALIAS = {
        'buy_and_hold':               'Benchmark 1',
        'moving_average_crossover':   'Benchmark 2',
        'threshold_only':             'Strategy 1',
        'mean_variance_scaling_0.5': 'Strategy 2a',
        'mean_variance_scaling_1': 'Strategy 2b',
        'mean_variance_scaling_1.5':  'Strategy 2c',
        'tail_risk_adjusted_confidence_1': 'Strategy 3a',
        'tail_risk_adjusted_confidence_2': 'Strategy 3b',
        'tail_risk_adjusted_confidence_3': 'Strategy 3c',
    }

    pnl_curves = {}
    metrics_list = []

    for key, params in strategy_grid:
        alias = STRATEGY_ALIAS[key]
        print(f"\n🔹 Testing {alias} ({key})")

        func_name = next(name for name in strategy_mapper.keys() if key.startswith(name))
        func = strategy_mapper[func_name]

        positions = func(df, **params)
        positions = np.clip(positions, -MAX_EXPOSURE, MAX_EXPOSURE)

        close = df['Close'].values
        returns = np.zeros_like(close)
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]

        gross_pnl = positions[:-1] * returns[1:]
        trades = np.abs(positions[1:] - positions[:-1])
        costs = trades * (FEE_RATE + SLIPPAGE_RATE)
        net_pnl = gross_pnl - costs

        account_value = np.array(INIT_CAPITAL * (1 + np.cumsum(net_pnl)))

        pnl_curves[alias] = account_value

        
        valid_idx = np.where(~np.isnan(account_value))[0]
        if len(valid_idx) > 0:
            last_valid_value = account_value[valid_idx[-1]]
            total_return = last_valid_value / INIT_CAPITAL - 1
        else:
            total_return = 0

        hours_per_year = 24 * 365
        hours_in_test = len(net_pnl)
        annual_factor = np.sqrt(hours_per_year / hours_in_test)
        sharpe = (np.mean(net_pnl) / (np.std(net_pnl) + 1e-8)) * annual_factor
        
        if len(valid_idx) > 0:
            valid_account_value = account_value[valid_idx]
            max_drawdown = np.max(np.maximum.accumulate(valid_account_value) - valid_account_value)
        else:
            max_drawdown = 0
        drawdown_ratio = max_drawdown / INIT_CAPITAL
        win_rate = np.mean(net_pnl > 0)

        metrics_list.append([
            alias,
            round(total_return, 4),
            round(sharpe, 4),
            round(drawdown_ratio, 4),
            round(win_rate, 4)
        ])

    df_metrics = pd.DataFrame(metrics_list,
                              columns=['Strategy','Total Return','Sharpe','Max Drawdown','Win Rate'])
    df_metrics.to_csv(os.path.join(save_dir, f"{model_tag}_performance.csv"), index=False)

    final_values = {}

    for alias, curve in pnl_curves.items():
        valid_idx = np.where(~np.isnan(curve))[0]
        if len(valid_idx) > 0:
            last_valid_value = curve[valid_idx[-1]]
        else:
            last_valid_value = INIT_CAPITAL  # 如果实在没有，默认初始值
        final_values[alias] = last_valid_value
        
    top3_aliases = sorted(final_values, key=final_values.get, reverse=True)[:3]

    cmap = cm.get_cmap('tab20', len(pnl_curves))
    plt.figure(figsize=(12, 7))

    for idx, (alias, curve) in enumerate(pnl_curves.items()):
        n = len(curve)
        timestamps = df['timestamp'].values[-n:]
        plt.plot(timestamps, curve, label=alias, color=cmap(idx))

    top3_text = f"Top 1: {top3_aliases[0]}\nTop 2: {top3_aliases[1]}\nTop 3: {top3_aliases[2]}"
    plt.text(0.01, 0.01, top3_text, transform=plt.gca().transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.7))

    plt.title(f"Account Value Curve – {model_tag}")
    plt.xlabel("Timestamp")
    plt.ylabel("Account Value ($)")
    plt.legend(loc='upper left', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_tag}_pnl_curve.png"), dpi=300)
    plt.close()

    print(f"✅ Saved results to {save_dir}")


if __name__ == "__main__":
    tasks = [
        ("strategy_experiments/results/gold_tf_prediction.csv", "strategy_experiments/results/gold_tf", "gold_tf"),
        ("strategy_experiments/results/gold_lstm_prediction.csv", "strategy_experiments/results/gold_lstm", "gold_lstm"),
        ("strategy_experiments/results/oil_tf_prediction.csv", "strategy_experiments/results/oil_tf", "oil_tf"),
        ("strategy_experiments/results/oil_lstm_prediction.csv", "strategy_experiments/results/oil_lstm", "oil_lstm"),
    ]
    active_hours = None
    active_hours = list(range(0, 22))
    for pred_path, save_dir, model_tag in tasks:
        run_backtest(pred_path, save_dir, model_tag, active_hours)