"""Garante que nenhuma corrida aparece em treino e teste do mesmo fold e que os tipos ficam balanceados."""
import numpy as np
import pandas as pd

from pdm import load_config
from pdm.cv import make_folds


def fake_data(n_corridas=200, per=5, seed=0):
    rng = np.random.default_rng(seed)
    tipos = rng.choice(["BH", "IF", "HSLA", "DP-TRIP"], n_corridas, p=[0.35, 0.2, 0.3, 0.15])
    return pd.DataFrame({
        "corrida_id": np.repeat([f"c{i}" for i in range(n_corridas)], per),
        "tipo": np.repeat(tipos, per),
    })


def test_corrida_nunca_em_dois_conjuntos():
    df, cfg = fake_data(), load_config()
    for train_idx, test_idx in make_folds(df, cfg):
        assert not set(df.corrida_id.iloc[train_idx]) & set(df.corrida_id.iloc[test_idx])


def test_proporcao_de_tipos_parecida_em_cada_fold():
    df, cfg = fake_data(), load_config()
    total = df.tipo.value_counts(normalize=True)
    for _, test_idx in make_folds(df, cfg):
        fold = df.tipo.iloc[test_idx].value_counts(normalize=True)
        assert (fold.reindex(total.index, fill_value=0) - total).abs().max() < 0.06
