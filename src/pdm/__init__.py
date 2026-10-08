"""PDM Remake: surrogate de propriedades mecânicas do aço + otimizador multiobjetivo."""
import random
from pathlib import Path

import numpy as np
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or PACKAGE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
