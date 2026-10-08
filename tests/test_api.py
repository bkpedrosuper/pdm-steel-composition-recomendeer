"""Smoke test da API. Precisa dos artefatos de `python -m pdm.train` (pula se não existirem)."""
import pytest
from fastapi.testclient import TestClient

from pdm import REPO_ROOT

pytestmark = pytest.mark.skipif(not (REPO_ROOT / "artifacts" / "model" / "model.pt").exists(),
                                reason="treine o modelo antes (python -m pdm.train)")


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_no_meio_do_dominio(client):
    from api.main import get_predictor
    p = get_predictor()
    mid = {c: (p.domain["min"][c] + p.domain["max"][c]) / 2 for c in p.inputs}
    body = client.post("/predict", json={"features": mid}).json()
    assert set(body["previsto"]) == {"LE", "LR", "AL"}


def test_recommend_avisa_fora_do_dominio(client):
    body = client.post("/recommend", json={"LE": 5000, "LR": 6000, "AL": 30}).json()
    assert body["avisos"] and body["recomendacoes"]
