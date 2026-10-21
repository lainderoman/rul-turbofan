"""Evaluation metrics for RUL prediction: RMSE and the NASA scoring function."""

import numpy as np


def rmse(y_true, y_pred):
    """Root Mean Squared Error, in cycles.

    RMSE = sqrt( mean( (y_pred - y_true)^2 ) )

    Early and late predictions are penalised in the same way.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def nasa_score(y_true, y_pred):
    """NASA scoring function from the PHM08 challenge (Saxena et al., 2008).

    For each engine, with d = y_pred - y_true (the prediction error):
        d < 0 (early prediction):  s = exp(-d / 13) - 1
        d >= 0 (late prediction):  s = exp( d / 10) - 1
    The final score is the SUM of s over all engines. Lower is better.

    Predicting failure too LATE is dangerous (the engine may fail before
    maintenance), so late errors are penalised more (divide by 10 instead
    of 13). The exponential also makes large errors very costly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # early prediction
        np.exp(d / 10) - 1,   # late prediction
    )
    return np.sum(scores)
