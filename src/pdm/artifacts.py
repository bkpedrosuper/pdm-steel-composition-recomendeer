"""Resolve onde está o modelo. Aceita três formas de URI:

    artifacts/model                        pasta local (saída do pdm.train)
    s3://bucket/.../model.tar.gz           saída do job da SageMaker
    models:/pdm-surrogate/3                versão registrada no MLflow
    models:/pdm-surrogate@champion         versão apontada por um alias no MLflow
"""
import os
import tarfile
import tempfile
from pathlib import Path

from pdm import REPO_ROOT

MODEL_NAME = "pdm-surrogate"   # nome do modelo no MLflow Model Registry


def tracking_uri() -> str:
    """MLflow em uso: $MLFLOW_TRACKING_URI ou o sqlite local do repo."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")


def describe_model_uri(uri: str | Path) -> str:
    """Texto legível do modelo: para models:/nome@alias, mostra qual versão o alias aponta."""
    uri = str(uri)
    if uri.startswith("models:/") and "@" in uri:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri())
        name, alias = uri.removeprefix("models:/").split("@", 1)
        return f"{name} v{MlflowClient().get_model_version_by_alias(name, alias).version} (@{alias})"
    return uri


def resolve_model_dir(uri: str | Path) -> Path:
    uri = str(uri)
    if uri.startswith("models:/"):
        return _from_mlflow(uri)
    if uri.startswith("s3://"):
        return _from_s3(uri)
    return Path(uri)


def _from_mlflow(uri: str) -> Path:
    import mlflow
    from mlflow.artifacts import download_artifacts

    mlflow.set_tracking_uri(tracking_uri())
    local = Path(download_artifacts(artifact_uri=uri))
    # o pdm.register guarda os arquivos do surrogate em artifacts/<nome da pasta de origem>/
    return next((local / "artifacts").rglob("model.pt")).parent


def _from_s3(uri: str) -> Path:
    import boto3

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    target = Path(tempfile.gettempdir()) / "pdm_model" / key.replace("/", "_")
    if not (target / "model.pt").exists():
        target.mkdir(parents=True, exist_ok=True)
        tar_path = target / "model.tar.gz"
        boto3.client("s3").download_file(bucket, key, str(tar_path))
        with tarfile.open(tar_path) as tar:
            tar.extractall(target, filter="data")
    return target
