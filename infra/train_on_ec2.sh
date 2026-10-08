#!/usr/bin/env bash
# Plano B do treino: roda o MESMO src/pdm/train.py na EC2 (dentro do container da API) e publica
# model.tar.gz no S3 no mesmo formato da SageMaker. Útil enquanto a cota de treino da SageMaker
# está em 0 (padrão de contas novas). Depois: bash infra/deploy_model.sh <uri impresso no fim>.
#
# Uso: bash infra/train_on_ec2.sh
set -euo pipefail
source infra/load_env.sh
source infra/.aws_outputs

JOB="ec2-train-$(date +%Y%m%d-%H%M%S)"
MODEL_URI="s3://$BUCKET/jobs/$JOB/output/model.tar.gz"

ssh -o StrictHostKeyChecking=accept-new -i infra/pdm-key.pem "ec2-user@$EC2_IP" bash -s <<EOF
set -euo pipefail
cd /opt/pdm
git pull --ff-only
mkdir -p data out
aws s3 cp "s3://$BUCKET/data/steel.parquet" data/steel.parquet --only-show-errors
docker compose -f docker-compose.aws.yml build api
docker compose -f docker-compose.aws.yml run --rm --no-deps \
  -v /opt/pdm/data:/data -v /opt/pdm/out:/out \
  api python -m pdm.train --data /data/steel.parquet --model-dir /out/model
tar -czf out/model.tar.gz -C out/model .
aws s3 cp out/model.tar.gz "$MODEL_URI" --only-show-errors
rm -rf out
EOF

echo "modelo: $MODEL_URI"
echo "próximo passo: bash infra/deploy_model.sh $MODEL_URI"
