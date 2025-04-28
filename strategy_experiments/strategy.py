# strategy.py

import pandas as pd
import numpy as np

def buy_and_hold(df):
    """
    Buy and hold strategy: always long 100%.
    """
    positions = np.ones(len(df))
    return positions

def moving_average_crossover(df):
    """
    Moving Average Crossover: Long when Close > EMA20, Short when Close < EMA20.
    """
    positions = np.where(df['Close'] > df['ema_20'], 1, -1)
    return positions

def threshold_only_trading(df):
    """
    Threshold-only Trading: Long if mu > 0, Short if mu < 0.
    """
    positions = np.where(df['mu'] > 0, 1, -1)
    return positions

def mean_variance_scaling(df, gamma=0.001):
    """
    Mean-Variance Scaling: Position size proportional to mu/variance.
    """
    variance = (df['sigma'] ** 2) * (df['nu'] / (df['nu'] - 2))
    positions = gamma * (df['mu'] / variance)
    positions = positions.clip(-1, 1)  # Clip position between -1 and 1
    return positions

def tail_risk_adjusted_confidence(df, gamma=0.01):
    """
    Tail-Risk Adjusted Confidence Trading:
    Positions scaled by mu, inversely by sigma and tail risk (nu).
    """
    effective_nu = np.maximum(df['nu'], 2.1)  # 防止nu接近2
    positions = gamma * (df['mu'] / (df['sigma'] * (effective_nu / (effective_nu - 2))))
    positions = positions.clip(-1, 1)  # 仓位clip在[-1, 1]
    return positions


# Strategy mapper for easy use
strategy_mapper = {
    'buy_and_hold': buy_and_hold,
    'moving_average_crossover': moving_average_crossover,
    'threshold_only': threshold_only_trading,
    'mean_variance_scaling': mean_variance_scaling,
    'tail_risk_adjusted_confidence': tail_risk_adjusted_confidence,
}
