#!/bin/bash
# User data da EC2 (Amazon Linux 2023). Os marcadores __X__ são preenchidos pelo setup_aws.sh.
set -euxo pipefail

# t3.micro tem só 1 GB de RAM: 2 GB de swap evitam falta de memória no build com PyTorch
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user

# plugin docker compose
mkdir -p /usr/local/lib/docker/cli-plugins
curl -sSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# o buildx do Amazon Linux é antigo demais para o `docker compose build` (exige >= 0.17)
curl -sSL https://api.github.com/repos/docker/buildx/releases/latest -o /tmp/buildx.json
BUILDX_VERSION=$(grep -m1 '"tag_name"' /tmp/buildx.json | cut -d'"' -f4)
curl -sSL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

git clone __REPO_URL__ /opt/pdm
cd /opt/pdm
# IP público da própria máquina (IMDSv2): origem que a UI do MLflow precisa aceitar
IMDS_TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
cat > .env <<EOF
PUBLIC_IP=$PUBLIC_IP
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
