"""Entrypoint de treino (local e SageMaker).

1. Validação cruzada (StratifiedGroupKFold) -> métricas honestas.
2. Modelo final treinado com todos os dados -> artefatos do surrogate.
3. Grava metrics.json junto do modelo. O registro no MLflow e na tabela `runs`
   é feito depois por `python -m pdm.register` (o job de treino não precisa de rede).

Local:      python -m pdm.train
SageMaker:  SM_CHANNEL_TRAIN e SM_MODEL_DIR são definidos pelo próprio job.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pdm import REPO_ROOT, load_config, set_seed
from pdm.cv import cross_validate
from pdm.trainer import Trainer


def find_data(path: str) -> Path:
    p = Path(path)
    return next(p.glob("*.parquet")) if p.is_dir() else p


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.environ.get("SM_CHANNEL_TRAIN",
                                                         str(REPO_ROOT / cfg["data"]["processed_path"])))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR",
                                                              str(REPO_ROOT / "artifacts" / "model")))
    args = parser.parse_args()

    set_seed(cfg["seed"])
    data_path = find_data(args.data)
    df = pd.read_parquet(data_path)
    print(f"dados: {data_path} ({len(df)} linhas, {df['corrida_id'].nunique()} corridas)")

    start = time.time()
    report, _ = cross_validate(df, cfg)
    metric_cols = ["r2", "mae", "mape", "acuracia_mae"]
    summary = pd.DataFrame(report.groupby(["tipo_aco", "saida"], sort=False)[metric_cols].mean()).reset_index()

    trainer = Trainer(cfg).fit(df)
    trainer.save(args.model_dir, df)
    elapsed = time.time() - start

    metrics = {
        "data": datetime.now(timezone.utc).isoformat(),
        "versao_dados": hashlib.sha256(data_path.read_bytes()).hexdigest()[:12],
        "tempo_treino": elapsed,
        "n_linhas": len(df),
        "params": {"n_folds": cfg["cv"]["n_folds"], "seed": cfg["seed"],
                   **{f"model_{k}": v for k, v in cfg["model"].items()},
                   **{f"train_{k}": v for k, v in cfg["train"].items()}},
        "cv_summary": summary.to_dict(orient="records"),
        "cv_folds": report.to_dict(orient="records"),
    }
    Path(args.model_dir, "metrics.json").write_text(json.dumps(metrics, indent=2, default=str),
                                                    encoding="utf-8")

    print("\nValidação cruzada, média dos folds (todos os aços):")
    print(summary.loc[summary["tipo_aco"] == "ALL"].round(3).to_string(index=False))
    print(f"\nmodelo + metrics.json em {args.model_dir} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
