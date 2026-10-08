"""Tabela `runs` com as métricas de cada treino (Postgres/RDS em produção, SQLite local).

Em produção, DATABASE_URL=postgresql+psycopg2://user:senha@host:5432/pdm; o Power BI lê dessa tabela.
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine

from pdm import REPO_ROOT

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), index=True),
    Column("data", DateTime),
    Column("tipo_aco", String(16)),
    Column("saida", String(8)),
    Column("r2", Float),
    Column("mae", Float),
    Column("mape", Float),
    Column("acuracia_mae", Float),
    Column("tempo_treino", Float),       # segundos do treino (CV + modelo final)
    Column("versao_dados", String(64)),  # hash do parquet
    Column("versao_modelo", String(64)), # run_id do MLflow
)


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        path = REPO_ROOT / "artifacts" / "metrics.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{Path(path).as_posix()}"
    return create_engine(url)


def save_run_metrics(rows: pd.DataFrame) -> int:
    engine = get_engine()
    metadata.create_all(engine)
    cols = [c.name for c in runs.columns if c.name != "id"]
    with engine.begin() as conn:
        conn.execute(runs.insert(), rows[cols].to_dict(orient="records"))
    return len(rows)
