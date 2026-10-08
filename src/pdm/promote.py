"""Promove uma versão do modelo: aponta o alias `champion` para ela no MLflow.

A API e o otimizador podem usar `models:/pdm-surrogate@champion` e passam a usar a nova
versão sem mudar código. Antes de promover, compare as métricas das versões no MLflow.

Uso:
    python -m pdm.promote 3                 # champion -> versão 3
    python -m pdm.promote 3 --alias challenger
"""
import argparse

import mlflow
from mlflow import MlflowClient

from pdm.artifacts import MODEL_NAME, tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="número da versão no Model Registry")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, args.alias, args.version)
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        tag = f"  <- @{args.alias}" if str(v.version) == str(args.version) else ""
        print(f"versão {v.version} (run {(v.run_id or '')[:8]}){tag}")


if __name__ == "__main__":
    main()
