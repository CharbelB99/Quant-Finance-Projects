import pandas as pd
import yfinance as yf

class DataHandler:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        """
        Fetches adjusted close prices from Yahoo Finance.
        Returns:
            prices (pd.DataFrame): dates x symbols
        """
        data = yf.download(self.symbols,
                           start=self.start_date,
                           end=self.end_date,
                           progress=False)
        prices = data['Adj Close'].copy()
        prices = prices.dropna(how='all')
        return prices

if __name__ == "__main__":
    dh = DataHandler(["SPY"], "2015-01-01", "2020-01-01")
    df = dh.fetch()
    print(df.head())