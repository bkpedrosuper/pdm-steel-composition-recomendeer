#!/usr/bin/env bash
# Registra um modelo treinado no MLflow de PRODUÇÃO como candidato (@challenger) e grava a
# tabela runs no RDS. A API continua servindo o @champion: nada muda para o usuário.
# Uso: bash infra/register_model.sh s3://pdm-remake-<conta>/jobs/<job>/output/model.tar.gz
set -euo pipefail
MODEL_URI="${1:?informe o s3://.../model.tar.gz}"
source infra/load_env.sh
source infra/.aws_outputs

ssh -o StrictHostKeyChecking=accept-new -i infra/pdm-key.pem "ec2-user@$EC2_IP" bash -s <<EOF
set -e
cd /opt/pdm
git pull -q --ff-only
docker compose -f docker-compose.aws.yml up -d --build api </dev/null 2>&1 | tail -1
docker compose -f docker-compose.aws.yml exec -T api python -m pdm.register "$MODEL_URI" --alias challenger </dev/null
EOF
echo "compare no MLflow: http://$EC2_IP:5000 (Models -> pdm-surrogate)"
echo "próximo passo: bash infra/promote.sh <versão>"
