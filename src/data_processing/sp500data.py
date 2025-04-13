import numpy as np
import pandas as pd
from typing import List


def load_sp500_tickers() -> List[str]:
    """ Loads S&P 500 tickers from Wikipedia. """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0].Symbol.to_list()

        tickers = [t.replace('.', '-').replace('/', '-')
                   for t in tickers]
        print(f"[DEBUG] Loaded {len(tickers)} tickers from Wikipedia.")
        print(
            f"[INFO] Removed META and FI Ticker due to a current bug in Polygon API adjusted stock prices.")
        tickers.remove('META')
        tickers.remove('FI')
        return tickers

    except Exception as e:
        print(
            f"[ERROR] Failed to load S&P 500 tickers from Wikipedia: {e}")
        return []
