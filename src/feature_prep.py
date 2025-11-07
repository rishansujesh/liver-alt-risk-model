"""Feature grouping and preprocessing utilities for NHANES liver project."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "elevated_alt"
ID_COLUMNS = {"SEQN"}
EXCLUDED_FEATURES = {TARGET_COLUMN, "LBXSATSI", *ID_COLUMNS}
CATEGORICAL_CANDIDATES = [
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC2",
    "DMDMARTZ",
    "SDDSRVYR",
    "DMDBORN4",
    "RIDEXMON",
]
DEMO_CANDIDATES = [
    "RIAGENDR",
    "RIDAGEYR",
    "RIDRETH1",
    "DMDEDUC2",
    "INDFMPIR",
    "DMDMARTZ",
    "SDDSRVYR",
    "DMDBORN4",
    "RIDEXMON",
]


def infer_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Infer feature groups from the provided dataframe.

    Returns a dictionary with keys ``categorical``, ``numeric``, ``alcohol``,
    ``demo``, ``labs``, and ``all``. Columns listed in ``EXCLUDED_FEATURES`` are
    never included in any group. Missing optional columns are skipped.
    """
    present_columns = [col for col in df.columns if col not in EXCLUDED_FEATURES]

    categorical_cols = [col for col in CATEGORICAL_CANDIDATES if col in present_columns]
    alcohol_cols = [col for col in present_columns if col.startswith("ALQ")]
    demo_cols = [col for col in DEMO_CANDIDATES if col in present_columns]
    labs_cols = [
        col for col in present_columns if col.startswith("LBX") and col != "LBXSATSI"
    ]

    numeric_cols = [
        col for col in present_columns if col not in set(categorical_cols)
    ]

    return {
        "categorical": categorical_cols,
        "numeric": numeric_cols,
        "alcohol": alcohol_cols,
        "demo": demo_cols,
        "labs": labs_cols,
        "all": present_columns,
    }


def build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    """Create a ColumnTransformer that handles categorical and numeric features."""
    transformers = []

    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("At least one of categorical or numeric features must be provided.")

    return ColumnTransformer(transformers=transformers)


__all__ = ["infer_column_groups", "build_preprocessor"]
