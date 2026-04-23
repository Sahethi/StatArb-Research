"""Yahoo Finance data source implementation."""
import pandas as pd
import yfinance as yf

from .base import DataSource


class YFinanceSource(DataSource):
    """
    Data source using the yfinance library.

    Fetches adjusted close prices and daily volume from Yahoo Finance.
    No credentials required.
    """

    def fetch_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers
        # Only forward-fill short gaps (1-2 days, e.g. exchange closures).
        # Long gaps (delistings, pre-IPO periods) stay NaN so the OU fitter
        # drops those days instead of generating spurious zero returns.
        prices = prices.ffill(limit=2)
        return prices

    def fetch_volume(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            volume = data["Volume"]
        else:
            volume = data[["Volume"]]
            volume.columns = tickers
        # Volume: treat missing as 0 (no trading that day) rather than bfill
        # so the volume-time adjustment doesn't project non-existent volume
        # back into pre-IPO dates.
        volume = volume.fillna(0)
        return volume
