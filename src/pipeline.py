"""The full preprocessing pipeline for one subset, as used in notebooks 04 and 05."""

import numpy as np

from src.data import (
    FD001_SENSORS,
    add_operating_condition,
    add_train_rul,
    apply_condition_scaler,
    fit_condition_scaler,
    load_cmapss,
    split_by_engine,
)
from src.sequences import make_last_windows, make_windows


def prepare_subset(subset, window_size=30, rul_cap=125, train_stride=1, split_seed=42):
    """Load one subset and return everything needed to train and evaluate the models.

    Steps (the same as in notebook 04):
    1. load the data, compute the RUL and cap it at `rul_cap`;
    2. add the operating condition of every row;
    3. split train / validation by engine;
    4. normalize the sensors per operating condition, fitted on the training engines only;
    5. build windows for train / validation and the last window of each test engine.

    Returns a dict with:
        train_part, val_part      normalized data frames (one row per cycle)
        X_train, y_train          training windows and their capped RUL
        X_val, y_val              validation windows and their capped RUL
        X_test                    last window of each test engine
        X_test_rf                 last cycle of each test engine (for the Random Forest)
        y_test                    true (uncapped) RUL of each test engine
    """
    train, test, test_rul = load_cmapss(subset)

    # 1. RUL capped at rul_cap
    train = add_train_rul(train)
    train["RUL"] = np.minimum(train["RUL"], rul_cap)

    # 2. Operating condition of every row
    train = add_operating_condition(train)
    test = add_operating_condition(test)

    # 3. Split by engine
    train_part, val_part = split_by_engine(train, val_fraction=0.2, seed=split_seed)

    # 4. Per-condition normalization, fitted on the training engines only
    means, stds = fit_condition_scaler(train_part, FD001_SENSORS)
    train_part = apply_condition_scaler(train_part, FD001_SENSORS, means, stds)
    val_part = apply_condition_scaler(val_part, FD001_SENSORS, means, stds)
    test = apply_condition_scaler(test, FD001_SENSORS, means, stds)

    # 5. Windows (short test engines are padded inside make_last_windows)
    X_train, y_train = make_windows(train_part, FD001_SENSORS, window_size, stride=train_stride)
    X_val, y_val = make_windows(val_part, FD001_SENSORS, window_size)
    X_test, test_units = make_last_windows(test, FD001_SENSORS, window_size)
    assert (test_units == test_rul["unit"].values).all()

    # Last recorded cycle of each test engine, for the Random Forest
    test_last_cycle = test.groupby("unit").tail(1).sort_values("unit")

    return {
        "train_part": train_part,
        "val_part": val_part,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test,
        "X_test_rf": test_last_cycle[FD001_SENSORS],
        "y_test": test_rul["true_rul"].values,
    }
