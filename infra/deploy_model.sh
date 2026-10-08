#!/usr/bin/env bash
# Atalho para o PRIMEIRO deploy (ou quando não precisa comparar): registra e já promove.
# No dia a dia, prefira os dois passos com revisão no meio:
#   bash infra/register_model.sh <s3 uri>   ->  compara no MLflow  ->  bash infra/promote.sh <versão>
# Uso: bash infra/deploy_model.sh s3://pdm-remake-<conta>/jobs/<job>/output/model.tar.gz
set -euo pipefail
MODEL_URI="${1:?informe o s3://.../model.tar.gz}"
source infra/load_env.sh
source infra/.aws_outputs

out=$(bash infra/register_model.sh "$MODEL_URI" | tee /dev/stderr)
version=$(grep -oE 'versão [0-9]+' <<<"$out" | tail -1 | grep -oE '[0-9]+')
bash infra/promote.sh "$version"
echo "API: http://$EC2_IP:8000/docs | MLflow: http://$EC2_IP:5000"
