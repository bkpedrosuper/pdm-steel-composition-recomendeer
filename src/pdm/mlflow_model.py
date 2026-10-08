"""Empacota o surrogate como modelo MLflow (pyfunc), para versionar no Model Registry.

Depois de registrado, qualquer versão pode ser carregada assim:
    model = mlflow.pyfunc.load_model("models:/pdm-surrogate/1")
    model.predict(df_com_as_29_colunas)  # -> DataFrame com LE, LR, AL
"""
import pandas as pd
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModel


class SurrogatePyfunc(PythonModel):
    def load_context(self, context):
        from pdm.predictor import SurrogatePredictor

        self.predictor = SurrogatePredictor(context.artifacts["model_dir"])

    # a classe base do MLflow anota o retorno como None, embora a API espere as previsões
    def predict(self, context, model_input, params=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """model_input: DataFrame com as 29 colunas -> DataFrame com LE, LR, AL."""
        p = self.predictor
        x = pd.DataFrame(model_input).loc[:, p.inputs].to_numpy()
        return pd.DataFrame(p.predict(x), columns=p.targets)


# "models from code": o MLflow salva este arquivo em vez de serializar o objeto com pickle
set_model(SurrogatePyfunc())
