"""Registra um treino no MLflow e na tabela `runs` (lida pelo Power BI).

No MLflow: um run (params + métricas de CV) e uma NOVA VERSÃO do modelo `pdm-surrogate`
no Model Registry (v1, v2, ...). Na tabela runs: uma linha por tipo de aço x saída.

Uso:
    python -m pdm.register artifacts/model                       # treino local
    python -m pdm.register s3://bucket/jobs/<job>/output/model.tar.gz   # job da SageMaker

MLFLOW_TRACKING_URI: servidor MLflow (padrão: sqlite local, mlflow.db)
DATABASE_URL:        Postgres/RDS da tabela runs (padrão: sqlite local, artifacts/metrics.db)
"""
import argparse
import json

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc import log_model

from pdm import PACKAGE_DIR
from pdm.artifacts import MODEL_NAME, resolve_model_dir, tracking_uri
from pdm.db import save_run_metrics
from pdm.mlflow_model import SurrogatePyfunc


def input_example(model_dir) -> pd.DataFrame:
    """Uma linha no meio do domínio de treino: documenta o formato de entrada no MLflow."""
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    lo, hi = meta["domain"]["min"], meta["domain"]["max"]
    return pd.DataFrame([{c: (lo[c] + hi[c]) / 2 for c in meta["inputs"]}])


def register(model_uri: str) -> str:
    model_dir = resolve_model_dir(model_uri)
    m = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.DataFrame(m["cv_summary"])

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("pdm-surrogate")
    with mlflow.start_run() as run:
        mlflow.log_params({**m["params"], "n_linhas": m["n_linhas"], "versao_dados": m["versao_dados"],
                           "origem": str(model_uri)})
        mlflow.log_metric("tempo_treino_s", m["tempo_treino"])
        for r in summary.to_dict(orient="records"):
            for metric in ["r2", "mae", "mape", "acuracia_mae"]:
                mlflow.log_metric(f"cv_{r['tipo_aco']}_{r['saida']}_{metric}", r[metric])

        example = input_example(model_dir)
        wrapper = SurrogatePyfunc()
        wrapper.load_context(type("Ctx", (), {"artifacts": {"model_dir": str(model_dir)}})())
        info = log_model(
            name="model",
            python_model=str(PACKAGE_DIR / "mlflow_model.py"),
            artifacts={"model_dir": str(model_dir)},
            code_paths=[str(PACKAGE_DIR)],
            signature=infer_signature(example, wrapper.predict(None, example)),
            input_example=example,
            pip_requirements=["torch", "scikit-learn", "pandas", "joblib", "pyyaml"],
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version

    rows = summary.assign(run_id=run.info.run_id, data=pd.Timestamp(m["data"]).to_pydatetime(),
                          tempo_treino=m["tempo_treino"], versao_dados=m["versao_dados"],
                          versao_modelo=f"{MODEL_NAME}/v{version}")
    n = save_run_metrics(rows)
    print(f"\nrun {run.info.run_id} -> modelo {MODEL_NAME} versão {version}; {n} linhas na tabela runs")
    return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_uri", help="pasta local ou s3://.../model.tar.gz")
    register(parser.parse_args().model_uri)


if __name__ == "__main__":
    main()
