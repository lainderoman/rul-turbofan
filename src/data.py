"""Functions to load the NASA C-MAPSS turbofan dataset and compute the RUL target."""

from pathlib import Path

import pandas as pd

# Folder with the original C-MAPSS .txt files.
# It is built from the location of this file, so it works no matter
# where the notebook or script is run from.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# The .txt files have no header. Each row is one engine at one cycle:
#   column 1      -> engine id ("unit")
#   column 2      -> time in cycles
#   columns 3-5   -> 3 operational settings
#   columns 6-26  -> 21 sensor measurements
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS


def load_cmapss(subset="FD001", data_dir=DATA_DIR):
    """Load the train set, test set and true test RUL of one C-MAPSS subset.

    Parameters
    ----------
    subset : str
        Name of the subset: "FD001", "FD002", "FD003" or "FD004".
    data_dir : str or Path
        Folder that contains the raw .txt files.

    Returns
    -------
    train : DataFrame
        Full run-to-failure trajectories (each engine runs until it fails).
    test : DataFrame
        Trajectories that stop some time before failure.
    test_rul : DataFrame
        One row per test engine with its true RUL at the last test cycle.
    """
    data_dir = Path(data_dir)

    # Columns are separated by one or more spaces, so we use a regex separator.
    # The lines also end with trailing spaces, which would create two empty
    # extra columns; "usecols" keeps only the 26 real columns.
    train = pd.read_csv(
        data_dir / f"train_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        usecols=range(len(COLUMN_NAMES)),
    )
    test = pd.read_csv(
        data_dir / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        usecols=range(len(COLUMN_NAMES)),
    )

    # The RUL file has one number per line: line i is the true RUL of test engine i.
    test_rul = pd.read_csv(
        data_dir / f"RUL_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=["true_rul"],
    )
    # Engine ids start at 1, in the same order as the lines of the file.
    test_rul["unit"] = range(1, len(test_rul) + 1)
    test_rul = test_rul[["unit", "true_rul"]]

    return train, test, test_rul


def add_train_rul(train):
    """Add a "RUL" column to the training data.

    In the training set every engine runs until failure, so its last cycle
    is the failure cycle. The RUL at any cycle is therefore:

        RUL = (last cycle of this engine) - (current cycle)

    which is 0 at the failure cycle.
    """
    train = train.copy()
    # For every row, the last cycle of the engine that row belongs to.
    last_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = last_cycle - train["cycle"]
    return train


def add_test_rul(test, test_rul):
    """Add a "RUL" column to the test data.

    A test engine stops before failure. The RUL file tells us the true RUL
    at its last recorded cycle, so for any earlier cycle:

        RUL = true_rul + (last recorded cycle) - (current cycle)
    """
    test = test.copy()
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    # Bring the true RUL of each engine onto every row of that engine.
    test = test.merge(test_rul, on="unit", how="left")
    test["RUL"] = test["true_rul"] + last_cycle - test["cycle"]
    test = test.drop(columns="true_rul")
    return test


def find_constant_columns(df, columns):
    """Return the columns that take a single value in the whole data frame.

    Such columns never change, so they carry no information about engine
    degradation and can be removed.
    """
    constant = []
    for col in columns:
        if df[col].nunique() == 1:
            constant.append(col)
    return constant
