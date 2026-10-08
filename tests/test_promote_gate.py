"""Gate de promoção: candidata pior que o champion além da tolerância é reprovada."""
from types import SimpleNamespace
from typing import cast

from mlflow import MlflowClient

from pdm.promote import gate


class FakeClient:
    """Imita o MlflowClient: cada versão aponta para um run com métricas de CV."""

    def __init__(self, mae_by_version: dict[str, dict[str, float]]):
        self.mae = mae_by_version

    def get_model_version(self, name, version):
        return SimpleNamespace(run_id=f"run-{version}")

    def get_run(self, run_id):
        version = run_id.removeprefix("run-")
        metrics = {f"cv_ALL_{t}_mae": v for t, v in self.mae[version].items()}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))


CHAMPION = {"LE": 10.0, "LR": 7.0, "AL": 1.40}


def test_candidata_melhor_e_aprovada():
    client = cast(MlflowClient, FakeClient({"1": CHAMPION, "2": {"LE": 9.5, "LR": 6.9, "AL": 1.38}}))
    assert gate(client, "2", "1", tolerance=0.02)


def test_piora_dentro_da_tolerancia_e_aprovada():
    client = cast(MlflowClient, FakeClient({"1": CHAMPION, "2": {"LE": 10.1, "LR": 7.0, "AL": 1.40}}))  # +1% no LE
    assert gate(client, "2", "1", tolerance=0.02)


def test_piora_alem_da_tolerancia_em_uma_saida_e_reprovada():
    client = cast(MlflowClient, FakeClient({"1": CHAMPION, "2": {"LE": 9.0, "LR": 6.5, "AL": 1.50}}))  # AL +7%
    assert not gate(client, "2", "1", tolerance=0.02)
