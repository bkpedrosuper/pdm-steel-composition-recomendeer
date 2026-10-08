"""Registra um treino: MLflow (params, métricas, artefatos) + tabela `runs` (lida pelo Power BI).

Uso:
    python -m pdm.register artifacts/model                       # treino local
    python -m pdm.register s3://bucket/jobs/<job>/output/model.tar.gz   # job da SageMaker

MLFLOW_TRACKING_URI: servidor MLflow (padrão: sqlite local)
DATABASE_URL:        Postgres/RDS da tabela runs (padrão: sqlite local)
"""
import argparse
import json
import os

import mlflow
import pandas as pd

from pdm import REPO_ROOT
from pdm.artifacts import resolve_model_dir
from pdm.db import save_run_metrics


def register(model_uri: str) -> str:
    model_dir = resolve_model_dir(model_uri)
    m = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.DataFrame(m["cv_summary"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                           f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"))
    mlflow.set_experiment("pdm-surrogate")
    with mlflow.start_run() as run:
        mlflow.log_params({**m["params"], "n_linhas": m["n_linhas"], "versao_dados": m["versao_dados"],
                           "origem": model_uri})
        mlflow.log_metric("tempo_treino_s", m["tempo_treino"])
        for r in summary.itertuples():
            for metric in ["r2", "mae", "mape", "acuracia_mae"]:
                mlflow.log_metric(f"cv_{r.tipo_aco}_{r.saida}_{metric}", getattr(r, metric))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

    rows = summary.assign(run_id=run.info.run_id, data=pd.Timestamp(m["data"]).to_pydatetime(),
                          tempo_treino=m["tempo_treino"], versao_dados=m["versao_dados"],
                          versao_modelo=run.info.run_id)
    n = save_run_metrics(rows)
    print(f"run {run.info.run_id} registrado no MLflow; {n} linhas na tabela runs")
    return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_uri", help="pasta local ou s3://.../model.tar.gz")
    register(parser.parse_args().model_uri)


if __name__ == "__main__":
    main()
