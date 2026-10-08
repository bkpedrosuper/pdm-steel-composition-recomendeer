#!/usr/bin/env bash
# Carrega o arquivo .env da raiz do repo (usado por setup_aws.sh, deploy_model.sh e teardown.sh).
# Faça `source infra/load_env.sh`; não precisa rodar direto.

export AWS_PAGER=""        # evita o erro do "more" da AWS CLI no Git Bash
export MSYS_NO_PATHCONV=1  # impede o Git Bash de trocar /aws/... por C:/Program Files/Git/aws/...

if [ -f .env ]; then
  # tr -d '\r' tira o fim de linha do Windows, que quebraria os valores
  set -a
  # shellcheck disable=SC1090
  source <(tr -d '\r' < .env)
  set +a
fi
