"""Métricas de regressão por saída: R², MAE, MAPE e acurácia baseada em MAE."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def accuracy_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - MAE / média do valor real. Ex.: reais [20,25,30], previstos [21,23,29] -> 0.9467."""
    return 1.0 - mean_absolute_error(y_true, y_pred) / np.mean(y_true)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(targets):
        t, p = y_true[:, i], y_pred[:, i]
        rows.append({
            "saida": name,
            "r2": r2_score(t, p),
            "mae": mean_absolute_error(t, p),
            "mape": mean_absolute_percentage_error(t, p) * 100,
            "acuracia_mae": accuracy_mae(t, p) * 100,
        })
    return pd.DataFrame(rows)


def report_by_type(df: pd.DataFrame, pred: np.ndarray, targets: list[str]) -> pd.DataFrame:
    """Relatório geral (tipo_aco = ALL) e por tipo de aço."""
    y = df.loc[:, targets].to_numpy()
    parts = [regression_report(y, pred, targets).assign(tipo_aco="ALL")]
    for tipo, idx in df.groupby("tipo").indices.items():
        parts.append(regression_report(y[idx], pred[idx], targets).assign(tipo_aco=str(tipo)))
    return pd.concat(parts, ignore_index=True)
