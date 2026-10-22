"""Simulate a maintenance policy on the test engines, with checks every few cycles."""

import numpy as np
import pandas as pd


def build_checks(test_part, feature_columns, window_size, interval):
    """Create the inspection checks of every test engine and the input window of each check.

    A test engine stops at its last recorded cycle, which we treat as "today's
    check". The earlier checks happened every `interval` cycles before it:
    last, last - interval, last - 2*interval, ... (only checks after the engine
    has flown at least `interval` cycles).

    At each check the model only sees the cycles up to that check. If fewer
    than `window_size` cycles are available, the window is padded by
    repeating the first cycle (as in make_last_windows).

    Returns
    -------
    X : array (number of checks, window_size, number of features)
    checks : DataFrame with one row per check: unit, cycle, true_rul
    """
    X_list = []
    rows = []

    for unit in np.sort(test_part["unit"].unique()):
        engine = test_part[test_part["unit"] == unit].sort_values("cycle")
        features = engine[feature_columns].values
        true_rul = engine["RUL"].values
        last_cycle = len(engine)

        # Check cycles, counted backwards from the last one, then put in time order
        check_cycles = list(range(last_cycle, interval - 1, -interval))
        check_cycles.reverse()

        for cycle in check_cycles:
            seen = features[:cycle]  # only the cycles up to this check
            if len(seen) < window_size:
                padding = np.repeat(seen[:1], window_size - len(seen), axis=0)
                seen = np.vstack([padding, seen])
            X_list.append(seen[-window_size:])
            rows.append({"unit": unit, "cycle": cycle, "true_rul": true_rul[cycle - 1]})

    X = np.array(X_list, dtype=np.float32)
    checks = pd.DataFrame(rows)
    return X, checks


def simulate_policy(checks, send, interval):
    """Apply a maintenance decision at every check, engine by engine, in time order.

    Parameters
    ----------
    checks : DataFrame from build_checks (unit, cycle, true_rul)
    send : boolean array, one value per check: True = send to maintenance
    interval : number of cycles between two checks

    For each engine we go through its checks in time order:
    - if the policy says "send": the engine is REMOVED. The wasted life is its
      true RUL at that moment. If the true RUL was >= interval, the engine would
      have safely reached the next check, so the removal was UNNECESSARY.
    - if the policy says "keep" but the true RUL is < interval: the engine fails
      before the next check -> MISSED FAILURE.
    - if no check triggers anything: the engine is still IN SERVICE at the end of
      the data (we do not know what happens after the last recorded cycle).

    Returns a DataFrame with one row per engine: unit, outcome, removal_cycle, wasted_life.
    """
    checks = checks.copy()
    checks["send"] = send
    results = []

    for unit, engine_checks in checks.groupby("unit"):
        outcome = "in service"
        removal_cycle = np.nan
        wasted_life = np.nan

        for _, check in engine_checks.iterrows():
            if check["send"]:
                outcome = "unnecessary removal" if check["true_rul"] >= interval else "necessary removal"
                removal_cycle = check["cycle"]
                wasted_life = check["true_rul"]
                break
            if check["true_rul"] < interval:
                outcome = "missed failure"
                break

        results.append({"unit": unit, "outcome": outcome,
                        "removal_cycle": removal_cycle, "wasted_life": wasted_life})

    return pd.DataFrame(results)


def summarize_outcomes(outcomes):
    """Count the outcomes and compute the wasted life of the removed engines."""
    counts = outcomes["outcome"].value_counts()
    removed = outcomes["wasted_life"].dropna()
    return {
        "missed failures": int(counts.get("missed failure", 0)),
        "necessary removals": int(counts.get("necessary removal", 0)),
        "unnecessary removals": int(counts.get("unnecessary removal", 0)),
        "in service": int(counts.get("in service", 0)),
        "avg wasted life": float(removed.mean()) if len(removed) > 0 else np.nan,
        "total wasted life": float(removed.sum()),
    }
