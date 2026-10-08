"""Recomenda composições: algoritmo evolutivo multiobjetivo usando o surrogate como avaliação.

Cada indivíduo é uma configuração com as 29 entradas (química + processo), limitadas ao
mínimo/máximo do treino. Os 3 objetivos são as distâncias relativas entre a propriedade
prevista e o alvo pedido pelo engenheiro (LE, LR, AL).

Uso:
    python -m pdm.optimize --LE 243 --LR 355 --AL 38.1                              # modelo local
    python -m pdm.optimize --LE 243 --LR 355 --AL 38.1 --model models:/pdm-surrogate/3
    python -m pdm.optimize --LE 243 --LR 355 --AL 38.1 --model models:/pdm-surrogate@champion
"""
import argparse

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from pdm import load_config
from pdm.predictor import SurrogatePredictor


class SteelDesignProblem(Problem):
    def __init__(self, predictor: SurrogatePredictor, target: np.ndarray):
        lo, hi = predictor.bounds()
        super().__init__(n_var=len(lo), n_obj=len(target), xl=lo, xu=hi)
        self.predictor, self.target = predictor, target

    def _evaluate(self, x, out, *args, **kwargs):
        pred = self.predictor.predict(x)
        out["F"] = np.abs(pred - self.target) / self.target   # erro relativo por propriedade


def recommend(predictor: SurrogatePredictor, target: dict[str, float], cfg: dict,
              algorithm: str | None = None) -> pd.DataFrame:
    ocfg = cfg["optimizer"]
    t = np.array([target[name] for name in predictor.targets], dtype=float)
    ref_dirs = get_reference_directions("das-dennis", len(t), n_partitions=ocfg["n_partitions"])

    algo = (algorithm or ocfg["algorithm"]).lower()
    if algo == "rvea":
        method = RVEA(ref_dirs=ref_dirs)
    elif algo == "nsga3":
        method = NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs))
    else:
        raise ValueError(f"algoritmo desconhecido: {algo}")

    res = minimize(SteelDesignProblem(predictor, t), method, ("n_gen", ocfg["n_gen"]),
                   seed=cfg["seed"], verbose=False)

    if res.X is None:
        raise RuntimeError("o otimizador não encontrou soluções")
    X = np.atleast_2d(res.X)
    pred = predictor.predict(X)
    out = pd.DataFrame(X, columns=predictor.inputs)
    for i, name in enumerate(predictor.targets):
        out[f"{name}_previsto"] = pred[:, i]
    out["erro_medio_pct"] = (np.abs(pred - t) / t).mean(axis=1) * 100
    # soluções do Pareto ordenadas pelo erro médio; o engenheiro escolhe entre as top-N
    return out.sort_values("erro_medio_pct").drop_duplicates().head(ocfg["top_n"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--LE", type=float, required=True)
    parser.add_argument("--LR", type=float, required=True)
    parser.add_argument("--AL", type=float, required=True)
    parser.add_argument("--algorithm", choices=["rvea", "nsga3"])
    parser.add_argument("--model", help="artifacts/model (padrão), models:/pdm-surrogate/3, "
                                        "models:/pdm-surrogate@champion ou s3://.../model.tar.gz")
    args = parser.parse_args()

    cfg = load_config()
    predictor = SurrogatePredictor(args.model)
    print(f"surrogate: {args.model or 'artifacts/model'}")
    target = {"LE": args.LE, "LR": args.LR, "AL": args.AL}
    for w in predictor.domain_check(target):
        print("AVISO:", w)

    recs = recommend(predictor, target, cfg, args.algorithm)
    cols = [f"{n}_previsto" for n in predictor.targets] + ["erro_medio_pct", "C", "Mn", "Si", "Nb", "Ti"]
    print(recs[cols].round(4).to_string())


if __name__ == "__main__":
    main()
