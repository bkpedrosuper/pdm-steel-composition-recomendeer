"""Surrogate pronto para uso: carrega os artefatos do treino e responde previsões."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from pdm import REPO_ROOT
from pdm.artifacts import resolve_model_dir
from pdm.model import MLP


class SurrogatePredictor:
    def __init__(self, model_uri: str | Path | None = None):
        """model_uri: pasta local ou s3://.../model.tar.gz (padrão: $PDM_MODEL_URI ou artifacts/model)."""
        model_dir = resolve_model_dir(model_uri or os.environ.get("PDM_MODEL_URI",
                                                                  REPO_ROOT / "artifacts" / "model"))
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.inputs: list[str] = self.meta["inputs"]
        self.targets: list[str] = self.meta["targets"]
        self.domain = self.meta["domain"]

        scalers = joblib.load(model_dir / "scalers.joblib")
        self.x_scaler, self.y_scaler = scalers["x"], scalers["y"]
        m = self.meta["model"]
        self.model = MLP(len(self.inputs), len(self.targets), m["hidden"], m["dropout"])
        self.model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (n, 29) na ordem de self.inputs -> (n, 3) com LE, LR, AL."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        with torch.no_grad():
            xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
            return self.y_scaler.inverse_transform(self.model(xt).numpy())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Limites de cada entrada vistos no treino (o otimizador não sai deles)."""
        lo = np.array([self.domain["min"][c] for c in self.inputs])
        hi = np.array([self.domain["max"][c] for c in self.inputs])
        return lo, hi

    def domain_check(self, values: dict[str, float]) -> list[str]:
        """Avisos para valores fora do mínimo/máximo do treino (extrapolação)."""
        warnings = []
        for name, v in values.items():
            lo, hi = self.domain["min"].get(name), self.domain["max"].get(name)
            if lo is not None and not lo <= v <= hi:
                warnings.append(f"{name}={v} fora do domínio de treino [{lo:g}, {hi:g}]: "
                                "previsão é extrapolação, use com cautela")
        return warnings
