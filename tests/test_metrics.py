import numpy as np

from pdm.metrics import accuracy_mae, regression_report


def test_accuracy_mae_exemplo_do_discurso():
    # reais [20,25,30], previstos [21,23,29]: MAE 1.333, média 25 -> 94.67%
    assert np.isclose(accuracy_mae(np.array([20, 25, 30]), np.array([21, 23, 29])), 1 - (4 / 3) / 25)


def test_report_previsao_perfeita():
    y = np.array([[200.0, 300.0, 40.0], [250.0, 350.0, 35.0], [300.0, 400.0, 30.0]])
    rep = regression_report(y, y, ["LE", "LR", "AL"])
    assert np.allclose(rep["r2"], 1) and np.allclose(rep["mae"], 0) and np.allclose(rep["acuracia_mae"], 100)
