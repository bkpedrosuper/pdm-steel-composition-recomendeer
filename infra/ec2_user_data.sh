#!/bin/bash
# User data da EC2 (Amazon Linux 2023). Os marcadores __X__ são preenchidos pelo setup_aws.sh.
set -euxo pipefail

dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

# plugin docker compose
mkdir -p /usr/local/lib/docker/cli-plugins
curl -sSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone __REPO_URL__ /opt/pdm
cd /opt/pdm
cat > .env <<EOF
BUCKET=__BUCKET__
RDS_HOST=__RDS_HOST__
DB_PASSWORD=__DB_PASSWORD__
AWS_DEFAULT_REGION=__REGION__
PDM_MODEL_URI=
EOF
chown -R ec2-user:ec2-user /opt/pdm

# banco separado para o MLflow (a nossa tabela runs fica no banco "pdm")
docker run --rm postgres:16 psql "postgresql://pdmadmin:__DB_PASSWORD__@__RDS_HOST__:5432/pdm" \
  -c "CREATE DATABASE mlflow" || true

# sobe só o MLflow; a API sobe depois do primeiro treino (infra/deploy_model.sh)
docker compose -f docker-compose.aws.yml up -d --build mlflow
