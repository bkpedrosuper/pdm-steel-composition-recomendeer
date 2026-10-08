"""Treino da MLP: escalonamento só no treino, Adam, early stopping por corridas separadas."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn

from pdm.model import MLP


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.inputs = cfg["columns"]["inputs"]
        self.targets = cfg["columns"]["targets"]
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: MLP | None = None
        self.history: list[dict] = []

    # ---------- treino ----------
    def _early_stop_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separa algumas corridas do treino para decidir quando parar (sem tocar no teste)."""
        gss = GroupShuffleSplit(n_splits=1, test_size=self.cfg["train"]["early_stop_frac"],
                                random_state=self.cfg["seed"])
        fit_idx, stop_idx = next(gss.split(df, groups=df[self.cfg["columns"]["group"]]))
        return df.iloc[fit_idx], df.iloc[stop_idx]

    def _tensors(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x_scaler.transform(df[self.inputs].values)
        y = self.y_scaler.transform(df[self.targets].values)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def fit(self, df: pd.DataFrame) -> "Trainer":
        tcfg = self.cfg["train"]
        fit_df, stop_df = self._early_stop_split(df)

        self.x_scaler.fit(fit_df.loc[:, self.inputs].to_numpy())
        self.y_scaler.fit(fit_df.loc[:, self.targets].to_numpy())
        x_fit, y_fit = self._tensors(fit_df)
        x_stop, y_stop = self._tensors(stop_df)

        mcfg = self.cfg["model"]
        model = MLP(len(self.inputs), len(self.targets), mcfg["hidden"], mcfg["dropout"])
        opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
        loss_fn = nn.MSELoss()

        best_loss, best_state, bad_epochs = np.inf, copy.deepcopy(model.state_dict()), 0
        for epoch in range(tcfg["max_epochs"]):
            model.train()
            for idx in torch.randperm(len(x_fit)).split(tcfg["batch_size"]):
                opt.zero_grad()
                loss = loss_fn(model(x_fit[idx]), y_fit[idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                stop_loss = loss_fn(model(x_stop), y_stop).item()
            self.history.append({"epoch": epoch, "stop_loss": stop_loss})

            if stop_loss < best_loss:
                best_loss, best_state, bad_epochs = stop_loss, copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= tcfg["patience"]:
                    break

        model.load_state_dict(best_state)
        self.model = model
        return self

    def _fitted(self) -> MLP:
        if self.model is None:
            raise RuntimeError("chame fit() antes de predict()/save()")
        return self.model

    # ---------- inferência ----------
    def predict(self, x: np.ndarray) -> np.ndarray:
        model = self._fitted()
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
            return self.y_scaler.inverse_transform(model(xt).numpy())

    # ---------- persistência ----------
    def save(self, out_dir: str | Path, train_df: pd.DataFrame) -> None:
        """Salva pesos, escalonadores e o domínio do treino (usado pelo otimizador e pelo aviso)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._fitted().state_dict(), out / "model.pt")
        joblib.dump({"x": self.x_scaler, "y": self.y_scaler}, out / "scalers.joblib")
        cols = self.inputs + self.targets
        domain = {"min": train_df[cols].min().to_dict(), "max": train_df[cols].max().to_dict()}
        meta = {"inputs": self.inputs, "targets": self.targets,
                "model": self.cfg["model"], "domain": domain}
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
