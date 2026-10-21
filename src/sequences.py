"""Build sliding windows over the sensor sequences, for the sequence models."""

import numpy as np


def make_windows(df, feature_columns, window_size, target_column="RUL", stride=1):
    """Cut every engine into overlapping windows of `window_size` cycles.

    For an engine with cycles 1..n and window_size = 30 we get the windows
    [1..30], [2..31], ..., [n-29..n]. The target of a window is the RUL at
    its LAST cycle, i.e. the model sees the last 30 cycles and predicts the
    RUL "now".

    With stride = k we keep only every k-th window, to make training faster.
    The first window is shifted so that the window ending at the last cycle
    (at failure) is always included. With stride = 1 nothing is skipped.

    Returns
    -------
    X : array of shape (number of windows, window_size, number of features)
    y : array of shape (number of windows,)
    """
    X_list = []
    y_list = []

    for unit in df["unit"].unique():
        engine = df[df["unit"] == unit]
        features = engine[feature_columns].values
        targets = engine[target_column].values

        # "end" is the index just after the last cycle of the window.
        # "first_end" is chosen so that, stepping by "stride", we land
        # exactly on the last cycle of the engine.
        first_end = window_size + (len(engine) - window_size) % stride
        for end in range(first_end, len(engine) + 1, stride):
            X_list.append(features[end - window_size:end])
            y_list.append(targets[end - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def make_last_windows(df, feature_columns, window_size):
    """Return the LAST window of every engine (one window per engine).

    This is what we use on the test set: the true RUL is only known at the
    last recorded cycle of each test engine.

    If an engine has fewer cycles than window_size, we pad the beginning by
    repeating its first cycle. (In FD001 every test engine has at least 31
    cycles, so with window_size = 30 no padding happens.)

    Returns
    -------
    X : array of shape (number of engines, window_size, number of features)
    units : array with the engine id of each window, in increasing order
    """
    X_list = []
    units = np.sort(df["unit"].unique())

    for unit in units:
        engine = df[df["unit"] == unit]
        features = engine[feature_columns].values

        if len(features) < window_size:
            n_missing = window_size - len(features)
            padding = np.repeat(features[:1], n_missing, axis=0)
            features = np.vstack([padding, features])

        X_list.append(features[-window_size:])

    X = np.array(X_list, dtype=np.float32)
    return X, units
