#!/usr/bin/env bash
# Registra um modelo treinado (MLflow + tabela runs no RDS) e (re)sobe a API com ele na EC2.
# Uso: bash infra/deploy_model.sh s3://pdm-remake-<conta>/jobs/<job>/output/model.tar.gz
set -euo pipefail
MODEL_URI="${1:?informe o s3://.../model.tar.gz}"
source infra/load_env.sh
source infra/.aws_outputs

ssh -o StrictHostKeyChecking=accept-new -i infra/pdm-key.pem "ec2-user@$EC2_IP" bash -s <<EOF
set -e
cd /opt/pdm
git pull --ff-only
sed -i "s|^PDM_MODEL_URI=.*|PDM_MODEL_URI=$MODEL_URI|" .env
docker compose -f docker-compose.aws.yml up -d --build api
docker compose -f docker-compose.aws.yml exec -T api python -m pdm.register "$MODEL_URI"
EOF
echo "API: http://$EC2_IP:8000/docs | MLflow: http://$EC2_IP:5000"
