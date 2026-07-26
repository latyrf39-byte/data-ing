import pandas as pd

from typing import Optional, Sequence, Any
from urllib.parse import urlparse
from pathlib import PurePosixPath


def load_data(
    url: str,
    *,
    sep: Optional[str] = None,
    header: Optional[int | str] = "infer",
    names: Optional[Sequence[str]] = None,
    **read_kwargs: Any
) -> pd.DataFrame:
    """
    Charge un dataset depuis une URL ou un fichier local.

    Formats supportés :
    - CSV, TXT, DATA, DAT
    - TSV
    - JSON, JSONL
    - Excel (.xlsx, .xls)
    - HTML (première table trouvée)
    - Parquet
    - Feather
    - Pickle
    """

    if not isinstance(url, str) or not url.strip():
        raise TypeError("url doit être une chaîne non vide.")

    path = urlparse(url).path
    suffixes = [s.lower() for s in PurePosixPath(path).suffixes]
    joined = "".join(suffixes)

    def has(*patterns: str) -> bool:
        return any(p in joined for p in patterns)

    def _read_csv(u: str, sep_arg: Optional[str]):
        csv_kwargs = dict(
            header=header,
            names=names,
            **read_kwargs
        )

        if sep_arg is None:
            return pd.read_csv(
                u,
                sep=None,
                engine="python",
                **csv_kwargs
            )

        return pd.read_csv(
            u,
            sep=sep_arg,
            low_memory=False,
            **csv_kwargs
        )

    try:

        # ===== Formats connus =====

        if has(".csv", ".txt", ".data", ".dat"):
            return _read_csv(url, sep)

        if has(".tsv"):
            return _read_csv(url, "\t")

        if has(".xlsx", ".xls"):
            return pd.read_excel(
                url,
                header=0 if header == "infer" else header,
                **read_kwargs
            )

        if has(".jsonl"):
            return pd.read_json(
                url,
                lines=True,
                **read_kwargs
            )

        if has(".json"):
            return pd.read_json(
                url,
                **read_kwargs
            )

        if has(".html", ".htm"):
            tables = pd.read_html(url)

            if len(tables) == 0:
                raise ValueError("Aucune table HTML trouvée.")

            return tables[0]

        if has(".parquet"):
            return pd.read_parquet(
                url,
                **read_kwargs
            )

        if has(".feather"):
            return pd.read_feather(
                url,
                **read_kwargs
            )

        if has(".pkl", ".pickle"):
            return pd.read_pickle(
                url
            )

        # ===== Fallbacks =====

        # CSV
        try:
            return _read_csv(url, None)
        except Exception:
            pass

        # TSV ou données séparées par espaces
        try:
            return pd.read_csv(
                url,
                sep=r"\s+",
                engine="python",
                header=header,
                names=names,
                **read_kwargs
            )
        except Exception:
            pass

        # JSONL puis JSON
        for kw in (dict(lines=True), dict()):
            try:
                return pd.read_json(
                    url,
                    **kw,
                    **read_kwargs
                )
            except Exception:
                continue

        # HTML
        try:
            tables = pd.read_html(url)

            if len(tables) > 0:
                return tables[0]
        except Exception:
            pass

        # Excel
        try:
            return pd.read_excel(
                url,
                header=0 if header == "infer" else header,
                **read_kwargs
            )
        except Exception:
            pass

        raise ValueError(
            "Format non reconnu ou non supporté."
        )

    except ImportError as e:
        raise ImportError(
            f"Moteur manquant : {e}. "
            "Installez les dépendances nécessaires "
            "(openpyxl, pyarrow, fastparquet...)."
        ) from e

    except Exception as e:
        raise ValueError(
            f"Échec de lecture de '{url}' : {e}"
        ) from e