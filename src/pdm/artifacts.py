"""Resolve onde está o modelo: pasta local ou model.tar.gz no S3 (saída do job da SageMaker)."""
import tarfile
import tempfile
from pathlib import Path


def resolve_model_dir(uri: str | Path) -> Path:
    uri = str(uri)
    if not uri.startswith("s3://"):
        return Path(uri)

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
