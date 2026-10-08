"""Validação cruzada sem vazamento: StratifiedGroupKFold (grupo = corrida, estrato = tipo de aço)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pdm.metrics import report_by_type
from pdm.trainer import Trainer


def make_folds(df: pd.DataFrame, cfg: dict):
    cols = cfg["columns"]
    sgkf = StratifiedGroupKFold(n_splits=cfg["cv"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    return list(sgkf.split(df, y=df[cols["strata"]], groups=df[cols["group"]]))


def cross_validate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Retorna métricas por fold/saída/tipo e as previsões out-of-fold."""
    targets = cfg["columns"]["targets"]
    oof = np.zeros((len(df), len(targets)))
    reports = []
    for k, (train_idx, test_idx) in enumerate(make_folds(df, cfg)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        trainer = Trainer(cfg).fit(train_df)
        pred = trainer.predict(test_df[cfg["columns"]["inputs"]].values)
        oof[test_idx] = pred
        rep = report_by_type(test_df, pred, targets).assign(fold=k)
        reports.append(rep)
        overall = rep[rep.tipo_aco == "ALL"].set_index("saida")["r2"].round(3).to_dict()
        print(f"fold {k}: {len(train_df)} treino / {len(test_df)} teste, "
              f"épocas={len(trainer.history)}, R²={overall}")
    return pd.concat(reports, ignore_index=True), oof
