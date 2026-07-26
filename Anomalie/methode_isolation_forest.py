import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    columns: list,
    contamination: float = 0.05,
    random_state: int = 42
):
    """
    Détection des anomalies avec Isolation Forest.

    Parameters
    ----------
    df : DataFrame
    columns : list
    contamination : float

    Returns
    -------
    DataFrame
    """

    df_copy = df.copy()

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(
        df_copy[columns]
    )

    df_copy["anomalie_iforest"] = (
        predictions == -1
    )

    return df_copy