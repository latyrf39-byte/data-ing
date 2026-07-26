import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

def load_neon(query):
    load_dotenv()

    engine = create_engine(
        os.getenv("DATABASE_URL")
    )

    return pd.read_sql(query, engine)