import numpy as np
from sklearn.metrics import auc
from glum import TweedieDistribution
import pandas as pd

def compute_bias(actual, preds, weights):
    return np.sum((preds - actual) * weights) / np.sum(weights)

def compute_rmse(actual, preds, weights):
    return np.sqrt(np.sum((actual - preds)**2  * weights)) / np.sum(weights)

def compute_mae(actual, preds, weights):
    return np.sum(np.abs(actual - preds) * weights) / np.sum(weights)

def compute_deviance(actual, preds, weights):
    TweedieDist = TweedieDistribution(1.5)
    return TweedieDist.deviance(actual, preds, sample_weight=weights) / np.sum(weights)

def compute_gini(actual, preds, weights):
    def lorenz_curve(y_true, y_pred, y_weights):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        y_weights = np.asarray(y_weights)

        # order samples by increasing predicted risk:
        ranking = np.argsort(y_pred)
        ranked_weights = y_weights[ranking]
        ranked_pure_premium = y_true[ranking]
        cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_weights)
        cumulated_claim_amount /= cumulated_claim_amount[-1]
        cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
        return cumulated_samples, cumulated_claim_amount
    
    ordered_samples, cum_claims = lorenz_curve(
        actual, preds, weights
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    return gini

def evaluate_metrics(actual, preds, weights):
    bias = compute_bias(actual, preds, weights)
    rmse = compute_rmse(actual, preds, weights)
    mae = compute_mae(actual, preds, weights)
    dev = compute_deviance(actual, preds, weights)
    gini = compute_gini(actual, preds, weights)

    results = {
        "metrics": ["Bias", "RMSE", "MAE", "Deviance", "Gini"],
        "values": [bias, rmse, mae, dev, gini]
    }
    return pd.DataFrame(results).set_index("metrics")

