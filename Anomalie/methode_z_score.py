import pandas as pd
import numpy as np
from scipy.stats import zscore


def detect_anomalies_zscore(
    df: pd.DataFrame,
    columns: list,
    threshold: float = 3
):
    df_copy = df.copy()

    z_scores = np.abs(
        zscore(df_copy[columns])
    )

    df_copy["anomalie_zscore"] = (
        z_scores > threshold
    ).any(axis=1)

    return df_copy