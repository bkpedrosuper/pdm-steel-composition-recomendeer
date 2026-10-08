# Deploy na AWS

## 0. Pré-requisitos (uma vez)

1. Instale a **AWS CLI v2** (https://aws.amazon.com/cli/) e rode `aws configure` com uma chave de um usuário IAM seu (não use a conta root). Região: `us-east-1`.
2. Crie um **alerta de orçamento** (Billing → Budgets) de US$ 10.
3. Publique este repositório no GitHub (público e **sem os dados**, que o `.gitignore` já exclui). A EC2 clona o repo.

## 1. Infraestrutura

```bash
cp .env.example .env     # edite o .env: DB_PASSWORD (só letras e números, 12+) e REPO_URL
bash infra/setup_aws.sh
```
Cria o bucket `pdm-remake-<conta>`, os papéis IAM `pdm-sagemaker-role` e `pdm-ec2-role`, os security groups (portas 22, 5000 e 8000 abertas **só para o seu IP**, e o RDS acessível só pela EC2 e pelo seu IP), o RDS Postgres `pdm-db` e a EC2 `pdm-server`. Grava os endereços em `infra/.aws_outputs`. A EC2 leva ~5 min para subir o MLflow em `http://<EC2_IP>:5000`.

## 2. Dados → S3

```bash
python -m pdm.data --upload        # s3://pdm-remake-<conta>/data/steel.parquet
```

## 3. Treino na SageMaker

```bash
python infra/launch_sagemaker.py
```
Imagem oficial PyTorch CPU, `ml.m5.large`. Leva alguns minutos e gera `s3://.../jobs/<job>/output/model.tar.gz`, com `metrics.json` dentro.

## 4. Registro e API

```bash
bash infra/deploy_model.sh s3://pdm-remake-<conta>/jobs/<job>/output/model.tar.gz
```
Na EC2: aponta a API para o novo modelo, sobe o container e roda `pdm.register`, que cria um run no MLflow (backend no RDS, artefatos no S3) e grava as linhas na tabela `runs` do banco `pdm`.

Testes:
```bash
curl http://<EC2_IP>:8000/health
curl -X POST http://<EC2_IP>:8000/recommend -H 'Content-Type: application/json' -d '{"LE":355,"LR":243,"AL":38.1}'
psql "postgresql://pdmadmin:$DB_PASSWORD@<RDS_HOST>:5432/pdm" -c "select saida, tipo_aco, r2, acuracia_mae from runs"
```
Power BI: Obter dados → PostgreSQL → `<RDS_HOST>`, banco `pdm`, tabela `runs`.

## 5. Desligar

```bash
bash infra/teardown.sh
```

## Custos aproximados (us-east-1)

| Recurso | Preço | Observação |
|---|---|---|
| EC2 t3.micro | ~US$ 0,0104/h | free tier (750 h/mês no 1º ano); fora dele ~US$ 0,25/dia |
| RDS db.t4g.micro | ~US$ 0,016/h | free tier nos 12 primeiros meses |
| SageMaker ml.m5.large | ~US$ 0,115/h | cobra só os minutos do job |
| S3 | centavos | poucos MB |

Uma sessão de estudo de 1 dia custa menos de US$ 2. Rode o **teardown** ao terminar.

## Notas de segurança (projeto de estudo)

- A senha do RDS vai no user data da EC2. Em produção, use o Secrets Manager ou o SSM Parameter Store.
- O MLflow não tem autenticação, e a proteção vem do security group, que só aceita o seu IP. Se o seu IP mudar, adicione o novo no `pdm-ec2-sg`.
