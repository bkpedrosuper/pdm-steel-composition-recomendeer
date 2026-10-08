# PDM Remake: Steel Composition Recommender

Recriação simplificada de um projeto real de P&D siderúrgico: um **modelo surrogate** prevê as
propriedades mecânicas do aço (limite de escoamento **LE**, resistência **LR** e alongamento **AL**)
a partir de 29 variáveis de composição química e processo, e um **algoritmo evolutivo
multiobjetivo** (RVEA ou NSGA-III) usa esse surrogate para recomendar composições que atinjam
as propriedades pedidas pelo engenheiro.

> Os dados são de produção industrial e **não estão no repositório**. O código espera os CSVs
> originais em `../PDM/models/model0/` (configurável em `src/pdm/config.yaml`).

## Arquitetura

```
 dados (CSV) ─► pdm.data ─► S3 ─► SageMaker (pdm.train) ─► model.tar.gz no S3
                                                              │
                     EC2 ─ pdm.register ◄─────────────────────┘
                      │      ├─► MLflow (backend RDS Postgres, artefatos S3)
                      │      └─► tabela runs no RDS ─► Power BI
                      └─ FastAPI ─► /predict  (surrogate)
                                 └► /recommend (RVEA/NSGA-III + surrogate + aviso de domínio)
```

| Peça | Arquivo | O que faz |
|---|---|---|
| Dados | `src/pdm/data.py` | Junta dados e aux, normaliza o tipo de aço, anonimiza a corrida (hash) e gera o parquet |
| Modelo | `src/pdm/model.py` | MLP multi-saída em PyTorch (29 → 128 → 64 → 3) |
| Treino | `src/pdm/trainer.py` | Escalonadores ajustados só no treino, Adam, early stopping com corridas separadas |
| Validação | `src/pdm/cv.py` | `StratifiedGroupKFold`: a mesma corrida nunca cai em treino e teste, e a proporção de tipos de aço é mantida |
| Métricas | `src/pdm/metrics.py` | R², MAE, MAPE e acurácia = 1 − MAE/média, geral e por tipo de aço |
| Registro | `src/pdm/register.py` + `db.py` | MLflow e tabela `runs(run_id, data, tipo_aco, saida, r2, mae, mape, acuracia_mae, tempo_treino, versao_dados, versao_modelo)` |
| Surrogate | `src/pdm/predictor.py` | Carrega o modelo (pasta local ou S3), prevê e avisa quando o pedido está fora do domínio de treino |
| Otimizador | `src/pdm/optimize.py` | Problema pymoo: 29 variáveis limitadas ao domínio e 3 objetivos (erro relativo a LE, LR e AL) |
| API | `api/main.py` | FastAPI: `/health`, `/predict`, `/recommend` |

## Rodando local

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:.

python -m pdm.data                       # data/processed/steel.parquet
python -m pdm.train                      # CV + modelo final em artifacts/model
python -m pdm.register artifacts/model   # MLflow (mlflow.db) + tabela runs (artifacts/metrics.db)
python -m pdm.optimize --LE 355 --LR 243 --AL 38.1     # recomendação via RVEA
pytest

uvicorn api.main:app --reload            # http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Com Docker (imita a AWS: Postgres = RDS, MLflow, API): `docker compose up -d --build`.

## Rodando na AWS

Passo a passo e custos em [`infra/README_aws.md`](infra/README_aws.md). Em resumo:

```bash
bash infra/setup_aws.sh                    # S3, IAM, RDS, EC2 com MLflow
python -m pdm.data --upload                # dados para o S3
python infra/launch_sagemaker.py           # treino na SageMaker
bash infra/deploy_model.sh s3://.../model.tar.gz   # registra e sobe a API na EC2
bash infra/teardown.sh                     # apaga tudo
```

## Decisões

- **Sem vazamento:** amostras da mesma corrida são quase idênticas. Dividir por linha infla as métricas, então o split é por corrida e estratificado por tipo de aço.
- **Escopo do otimizador:** a busca fica dentro do mínimo e máximo de cada variável no treino. Se o alvo pedido estiver fora do domínio, o sistema avisa em vez de bloquear.
- **O treino não depende de rede:** o job da SageMaker só grava `metrics.json` junto do modelo, e o registro (MLflow e RDS) acontece dentro da VPC, na EC2. Assim nem o MLflow nem o RDS ficam expostos à internet.
- **Próximos passos:** ensemble de redes, arquivo de diversidade por entropia de Shannon no otimizador, retreino mensal agendado e monitoramento de drift (PSI).
