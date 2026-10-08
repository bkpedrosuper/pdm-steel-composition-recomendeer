"""Dispara o treino na SageMaker (script mode) usando só boto3.

Empacota src/ como sourcedir.tar.gz, envia ao S3 e cria um training job com a imagem
oficial de PyTorch (CPU). O container instala src/requirements.txt e roda sm_entry.py.

Uso (raiz do repo, após setup_aws.sh e `python -m pdm.data --upload`):
    python infra/launch_sagemaker.py
"""
import io
import tarfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = dict(line.split("=", 1) for line in (ROOT / "infra" / ".aws_outputs").read_text().split()
               if "=" in line)
REGION, BUCKET, ROLE = OUTPUTS["REGION"], OUTPUTS["BUCKET"], OUTPUTS["SAGEMAKER_ROLE"]
IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
INSTANCE = "ml.m5.large"


def upload_sourcedir(s3, job_name: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in (ROOT / "src").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=path.relative_to(ROOT / "src").as_posix())
    key = f"jobs/{job_name}/sourcedir.tar.gz"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    return f"s3://{BUCKET}/{key}"


def main() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)
    job_name = time.strftime("pdm-train-%Y%m%d-%H%M%S")
    source = upload_sourcedir(s3, job_name)

    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        HyperParameters={  # valores em JSON: é assim que o script mode lê
            "sagemaker_program": '"sm_entry.py"',
            "sagemaker_submit_directory": f'"{source}"',
            "sagemaker_region": f'"{REGION}"',
        },
        InputDataConfig=[{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{BUCKET}/data/",
                                            "S3DataDistributionType": "FullyReplicated"}},
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/jobs/"},
        ResourceConfig={"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )
    print(f"job {job_name} criado; acompanhando (logs no CloudWatch /aws/sagemaker/TrainingJobs)...")
    sm.get_waiter("training_job_completed_or_stopped").wait(
        TrainingJobName=job_name, WaiterConfig={"Delay": 30, "MaxAttempts": 120})

    desc = sm.describe_training_job(TrainingJobName=job_name)
    print("status:", desc["TrainingJobStatus"], desc.get("FailureReason", ""))
    if desc["TrainingJobStatus"] == "Completed":
        print("modelo:", desc["ModelArtifacts"]["S3ModelArtifacts"])
        print("próximo passo: bash infra/deploy_model.sh", desc["ModelArtifacts"]["S3ModelArtifacts"])


if __name__ == "__main__":
    main()
