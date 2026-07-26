import pandas as pd
import requests

def load(url):
    return pd.read_csv(url)