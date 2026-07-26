import pandas as pd
import re


def transform_data(df: pd.DataFrame):
    """
    Nettoyage et transformation générique d'un DataFrame.
    """

    rapport = {}

    # ==================================================
    # 1. Valeurs manquantes
    # ==================================================

    missing_before = df.isna().sum()

    rapport["missing_values"] = missing_before

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()

            if not mode.empty:
                df[col] = df[col].fillna(mode[0])

    # ==================================================
    # 2. Doublons
    # ==================================================

    nb_duplicates = df.duplicated().sum()

    rapport["duplicates_removed"] = nb_duplicates

    df = df.drop_duplicates()

    # ==================================================
    # 3. Snake Case
    # ==================================================

    mapping = {}

    for col in df.columns:

        new_col = re.sub(r"[^a-zA-Z0-9]+", "_", col)

        new_col = new_col.lower()

        new_col = new_col.strip("_")

        mapping[col] = new_col

    df = df.rename(columns=mapping)

    rapport["column_mapping"] = mapping

    # ==================================================
    # 4. Conversion des types
    # ==================================================

    for col in df.columns:

        try:
            df[col] = pd.to_numeric(df[col])

        except Exception:
            df[col] = df[col].astype("category")

    # ==================================================
    # 5. Variables dérivées
    # ==================================================

    if {"alcohol", "malic_acid"}.issubset(df.columns):

        df["alcohol_malic_ratio"] = (
            df["alcohol"] / df["malic_acid"]
        )

    if {"flavanoids", "total_phenols"}.issubset(df.columns):

        df["flavanoid_ratio"] = (
            df["flavanoids"]
            / df["total_phenols"]
        )

    # ==================================================
    # 6. Contrôles qualité
    # ==================================================

    if "alcohol" in df.columns:
        assert (df["alcohol"] >= 0).all(), \
            "Des valeurs négatives ont été détectées dans alcohol."

    if "malic_acid" in df.columns:
        assert (df["malic_acid"] >= 0).all(), \
            "Des valeurs négatives ont été détectées dans malic_acid."

    return df, rapport