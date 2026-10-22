"""Functions to load the NASA C-MAPSS turbofan dataset and compute the RUL target."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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

# The four C-MAPSS subsets
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# The 14 sensors kept for FD001 after the analysis in notebooks/01_eda.ipynb
# (constant sensors 1, 5, 10, 16, 18, 19 and quasi-constant sensor 6 removed).
# notebooks/04_all_subsets.ipynb uses the same 14 sensors for every subset.
FD001_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


# Physical meaning of the 21 sensors (Saxena et al., 2008, Table 2):
# sensor name -> (symbol, description)
SENSOR_DESCRIPTIONS = {
    "sensor_1": ("T2", "Total temperature at fan inlet"),
    "sensor_2": ("T24", "Total temperature at LPC outlet"),
    "sensor_3": ("T30", "Total temperature at HPC outlet"),
    "sensor_4": ("T50", "Total temperature at LPT outlet"),
    "sensor_5": ("P2", "Pressure at fan inlet"),
    "sensor_6": ("P15", "Total pressure in bypass duct"),
    "sensor_7": ("P30", "Total pressure at HPC outlet"),
    "sensor_8": ("Nf", "Physical fan speed"),
    "sensor_9": ("Nc", "Physical core speed"),
    "sensor_10": ("epr", "Engine pressure ratio (P50/P2)"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30"),
    "sensor_13": ("NRf", "Corrected fan speed"),
    "sensor_14": ("NRc", "Corrected core speed"),
    "sensor_15": ("BPR", "Bypass ratio"),
    "sensor_16": ("farB", "Burner fuel-air ratio"),
    "sensor_17": ("htBleed", "Bleed enthalpy"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "sensor_20": ("W31", "HPT coolant bleed"),
    "sensor_21": ("W32", "LPT coolant bleed"),
}


def sensor_label(sensor):
    """Short label for plots, e.g. "sensor_11" -> "s11 Ps30"."""
    symbol = SENSOR_DESCRIPTIONS[sensor][0]
    number = sensor.split("_")[1]
    return f"s{number} {symbol}"


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
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}, expected one of {SUBSETS}")
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


def split_by_engine(train, val_fraction=0.2, seed=42):
    """Split the training data into train / validation parts BY ENGINE.

    Every engine goes completely to one side, so no engine appears in both
    parts. With the same seed this gives the same engines as in
    notebooks/02_baselines.ipynb.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(train, groups=train["unit"]))
    return train.iloc[train_idx], train.iloc[val_idx]


def add_operating_condition(df):
    """Add a "condition" column that identifies the operating condition of each row.

    The 3 operational settings (altitude, Mach number, throttle angle) take
    only a few distinct values plus a tiny measurement noise. Rounding them
    removes the noise, and the combination of the 3 rounded values names
    the condition, e.g. "42_0.84_100". FD001 and FD003 have 1 condition,
    FD002 and FD004 have 6.
    """
    df = df.copy()
    # Adding 0.0 turns -0.0 into 0.0, so that tiny negative noise around
    # zero does not create a separate condition.
    setting_1 = df["setting_1"].round(0) + 0.0
    setting_2 = df["setting_2"].round(2) + 0.0
    setting_3 = df["setting_3"].round(0) + 0.0
    df["condition"] = (
        setting_1.map("{:g}".format) + "_"
        + setting_2.map("{:g}".format) + "_"
        + setting_3.map("{:g}".format)
    )
    return df


def fit_condition_scaler(train, columns):
    """Compute the mean and standard deviation of each column, per operating condition.

    Must be called on the TRAINING engines only, so that no information
    from validation or test engines is used.

    Returns
    -------
    means, stds : DataFrames with one row per condition and one column per sensor
    """
    means = train.groupby("condition")[columns].mean()
    stds = train.groupby("condition")[columns].std()
    return means, stds


def apply_condition_scaler(df, columns, means, stds):
    """Normalize every row with the mean and std of ITS OWN operating condition.

        normalized value = (value - mean of the condition) / std of the condition

    After this, a value of 0 means "average for this flight condition",
    so changes in the flight condition no longer hide the degradation trend.
    """
    df = df.copy()
    # Look up, for every row, the mean and std of its condition
    row_means = means.loc[df["condition"]].values
    row_stds = stds.loc[df["condition"]].values
    df[columns] = (df[columns].values - row_means) / row_stds
    return df


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
