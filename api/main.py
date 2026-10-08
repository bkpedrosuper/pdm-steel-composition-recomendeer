"""API do Steel Composition Recommender.

GET  /health     -> status
POST /predict    -> propriedades previstas para uma composição
POST /recommend  -> composições recomendadas para propriedades-alvo

Rodar: uvicorn api.main:app --reload
Modelo servido: $PDM_MODEL_URI (em produção, models:/pdm-surrogate@champion).
"""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pdm import REPO_ROOT, load_config
from pdm.artifacts import describe_model_uri
from pdm.optimize import recommend
from pdm.predictor import SurrogatePredictor

app = FastAPI(title="PDM Remake: Steel Composition Recommender")


MODEL_URI = os.environ.get("PDM_MODEL_URI", str(REPO_ROOT / "artifacts" / "model"))


@lru_cache
def get_predictor() -> SurrogatePredictor:
    # carregado uma vez por processo: depois de promover, recrie o container para recarregar
    return SurrogatePredictor(MODEL_URI)


@lru_cache
def served_model() -> str:
    return describe_model_uri(MODEL_URI)   # ex.: "pdm-surrogate v2 (@champion)"


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(..., description="as 29 entradas pelo nome da coluna")


class RecommendRequest(BaseModel):
    LE: float = Field(..., gt=0, description="limite de escoamento alvo (MPa)")
    LR: float = Field(..., gt=0, description="limite de resistência alvo (MPa)")
    AL: float = Field(..., gt=0, description="alongamento alvo (%)")
    algorithm: str = Field("rvea", pattern="^(rvea|nsga3)$")


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "modelo": served_model(), "inputs": len(p.inputs), "targets": p.targets}


@app.post("/predict")
def predict(req: PredictRequest):
    p = get_predictor()
    missing = [c for c in p.inputs if c not in req.features]
    if missing:
        raise HTTPException(422, f"faltam entradas: {missing}")
    pred = p.predict([[req.features[c] for c in p.inputs]])[0]
    return {"previsto": dict(zip(p.targets, pred.round(2).tolist())),
            "avisos": p.domain_check(req.features)}


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    p = get_predictor()
    target = {"LE": req.LE, "LR": req.LR, "AL": req.AL}
    recs = recommend(p, target, load_config(), req.algorithm)
    return {"alvo": target, "avisos": p.domain_check(target),
            "recomendacoes": recs.round(4).to_dict(orient="records")}
