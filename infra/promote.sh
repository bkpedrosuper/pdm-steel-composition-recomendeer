#!/usr/bin/env bash
# Promove uma versão a @champion em PRODUÇÃO e recarrega a API.
#   1. gate: compara o MAE de CV com o champion atual (pdm.promote) e aborta se piorar
#   2. aponta a API para models:/pdm-surrogate@champion e recria o container
#   3. confere no /health qual versão está sendo servida
#
# Uso:
#   bash infra/promote.sh 2              # promove a v2 se passar no gate
#   bash infra/promote.sh 1 --force      # rollback para a v1, ignorando o gate
#   bash infra/promote.sh 2 --dry-run    # só mostra a comparação
set -euo pipefail
VERSION="${1:?informe a versão}"; shift
EXTRA="$*"
source infra/load_env.sh
source infra/.aws_outputs

ssh -o StrictHostKeyChecking=accept-new -i infra/pdm-key.pem "ec2-user@$EC2_IP" bash -s <<EOF
set -e
cd /opt/pdm
git pull -q --ff-only
docker compose -f docker-compose.aws.yml exec -T api python -m pdm.promote $VERSION $EXTRA </dev/null
case " $EXTRA " in *" --dry-run "*) exit 0 ;; esac
sed -i "s|^PDM_MODEL_URI=.*|PDM_MODEL_URI=models:/pdm-surrogate@champion|" .env
docker compose -f docker-compose.aws.yml up -d --force-recreate api </dev/null 2>&1 | tail -1
EOF

case " $EXTRA " in *" --dry-run "*) exit 0 ;; esac
echo "aguardando a API recarregar..."
for _ in $(seq 1 60); do
  health=$(curl -s --max-time 5 "http://$EC2_IP:8000/health" || true)
  [[ "$health" == *'"status":"ok"'* ]] && { echo "$health"; exit 0; }
  sleep 3
done
echo "a API não respondeu em 3 min; veja: ssh ... 'docker logs pdm-api-1'"; exit 1
