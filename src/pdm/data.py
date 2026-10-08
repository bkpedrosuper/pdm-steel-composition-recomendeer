"""Prepara o dataset: junta dados + aux do model0, normaliza tipos e anonimiza corridas.

Uso:
    python -m pdm.data            # gera data/processed/steel.parquet
    python -m pdm.data --upload   # e envia para s3://$PDM_BUCKET/data/steel.parquet
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pdm import REPO_ROOT, load_config


class SteelData:
    """Carrega os CSVs originais e produz uma tabela limpa: entradas, saídas, tipo e corrida."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.raw_dir = (REPO_ROOT / cfg["data"]["raw_dir"]).resolve()
        self.out_path = REPO_ROOT / cfg["data"]["processed_path"]

    def _read_split(self, split: str) -> pd.DataFrame:
        data = pd.read_csv(self.raw_dir / f"df_{split}.csv")
        aux = pd.read_csv(self.raw_dir / f"aux_{split}.csv", encoding="latin1")
        # dados e aux foram exportados na mesma ordem; a coluna sem nome é o índice comum
        aligned = np.array_equal(data["Unnamed: 0"].to_numpy(), aux["Unnamed: 0"].to_numpy())
        assert aligned, "dados e aux desalinhados"
        return pd.concat([data, aux[["Corrida", "Tipo"]]], axis=1)

    @staticmethod
    def _anonymize(value: str) -> str:
        return hashlib.sha256(str(value).encode()).hexdigest()[:12]

    def build(self) -> pd.DataFrame:
        cols = self.cfg["columns"]
        df = pd.concat([self._read_split("train"), self._read_split("valid")], ignore_index=True)

        df = df.rename(columns=dict(zip(cols["raw_targets"], cols["targets"])))
        df["tipo"] = df["Tipo"].map(self.cfg["steel_type_map"])
        df["corrida_id"] = df["Corrida"].map(self._anonymize)

        df = df.loc[:, cols["inputs"] + cols["targets"] + ["tipo", "corrida_id"]]
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame) -> Path:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.out_path, index=False)
        return self.out_path


def upload_to_s3(path: Path, cfg: dict) -> str:
    import boto3

    outputs = REPO_ROOT / "infra" / ".aws_outputs"   # gerado pelo infra/setup_aws.sh
    if outputs.exists():
        os.environ.setdefault("PDM_BUCKET", dict(l.split("=", 1) for l in outputs.read_text().split()
                                                 if "=" in l)["BUCKET"])
    bucket = os.environ.get("PDM_BUCKET", cfg["aws"]["bucket"])
    key = cfg["aws"]["data_key"]
    boto3.client("s3", region_name=cfg["aws"]["region"]).upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="envia o parquet para o S3")
    args = parser.parse_args()

    cfg = load_config()
    steel = SteelData(cfg)
    df = steel.build()
    path = steel.save(df)

    print(f"{len(df)} linhas, {df['corrida_id'].nunique()} corridas -> {path}")
    print(df["tipo"].value_counts().to_string())
    if args.upload:
        print("enviado para", upload_to_s3(path, cfg))


if __name__ == "__main__":
    main()
