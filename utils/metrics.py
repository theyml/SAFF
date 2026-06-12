import numpy as np
from sklearn.metrics import mean_squared_log_error, r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MSLE(pred, true):
    return MSE(np.log1p(pred), np.log1p(true))

def RMSLE(pred, true):
    return np.sqrt(MSLE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def calculate_metrics(true, pred):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    rmsle = RMSLE(np.where(pred<0, 0, pred) , true)
    smape = SMAPE(pred, true)

    return mae, rmse, rmsle, smape

def calculate_results(y_true: np.ndarray, y_pred:np.ndarray):
    mae, rmse, rmsle, smape  = calculate_metrics(y_true, y_pred)
    
    return {
        'rmse': rmse, 'mae': mae, 
        'rmsle': rmsle, 'smape': smape
    }


def calculate_financial_metrics(last_close, pred_close, true_close, periods_per_year: int = 252, risk_aversion: float = 3.0):
    """
    Compute trading-style metrics from one-step-ahead close predictions.

    Assumptions:
    - signal = sign(predicted next-step return)
    - strategy return = signal * realized next-step return
    - no transaction cost
    - daily frequency by default (252 trading days / year)
    """
    last_close = np.asarray(last_close, dtype=np.float64)
    pred_close = np.asarray(pred_close, dtype=np.float64)
    true_close = np.asarray(true_close, dtype=np.float64)

    eps = 1e-12
    asset_ret = (true_close - last_close) / np.clip(last_close, eps, None)
    pred_ret = (pred_close - last_close) / np.clip(last_close, eps, None)

    signal = np.sign(pred_ret)
    strat_ret = signal * asset_ret

    equity_curve = np.cumprod(1.0 + strat_ret)
    cw = float(equity_curve[-1]) if len(equity_curve) > 0 else 1.0

    n = max(len(strat_ret), 1)
    apy = float(cw ** (periods_per_year / n) - 1.0)

    mean_r = float(np.mean(strat_ret)) if len(strat_ret) > 0 else 0.0
    var_r = float(np.var(strat_ret)) if len(strat_ret) > 0 else 0.0
    std_r = float(np.std(strat_ret)) if len(strat_ret) > 0 else 0.0

    cer = float(periods_per_year * (mean_r - 0.5 * risk_aversion * var_r))
    asr = float(np.sqrt(periods_per_year) * mean_r / (std_r + eps))

    downside = strat_ret[strat_ret < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    sor = float(np.sqrt(periods_per_year) * mean_r / (downside_std + eps))

    running_max = np.maximum.accumulate(equity_curve) if len(equity_curve) > 0 else np.array([1.0])
    drawdowns = 1.0 - equity_curve / np.clip(running_max, eps, None) if len(equity_curve) > 0 else np.array([0.0])
    md = float(np.max(drawdowns))
    cr = float(apy / (md + eps))
    avo = float(std_r * np.sqrt(periods_per_year))

    return {
        'CW': cw,
        'APY': apy,
        'CER': cer,
        'ASR': asr,
        'SoR': sor,
        'CR': cr,
        'MD': md,
        'AVO': avo,
    }
