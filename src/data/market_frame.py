from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(slots=True)
class MarketFrame:
    """
    Represents OHLCV market data from CSV file.

    The class intentionally does not compute technical indicators,
    labels, or machine learning features. Those belong to later stages
    of the data pipeline.
    """

    def __post_init__(self):

        if self.data.empty:
            raise EmptyDatasetException()

        self.data = self.data.sort_values(
            [self.ticker_column, self.datetime_column]
        )

        self.data.reset_index(drop=True, inplace=True)

        if self.data.duplicated(
                [self.ticker_column, self.datetime_column]
        ).any():
            raise DuplicateRowsException()


    data: pd.DataFrame

    ticker_column: str = "Ticker"
    datetime_column: str = "Date"

    open_column: str = "Open"
    high_column: str = "High"
    low_column: str = "Low"
    close_column: str = "Close"
    volume_column: str = "Volume"

    def copy(self) -> "MarketData":
        return MarketData(
            data=self.data.copy(deep=True),
            ticker_column=self.ticker_column,
            datetime_column=self.datetime_column,
            open_column=self.open_column,
            high_column=self.high_column,
            low_column=self.low_column,
            close_column=self.close_column,
            volume_column=self.volume_column,
        )

    @property
    def tickers(self) -> list[str]:
        return sorted(self.data[self.ticker_column].unique())

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def start_date(self):
        return self.data[self.datetime_column].min()

    @property
    def end_date(self):
        return self.data[self.datetime_column].max()

    @property
    def date_range(self):
        return (
            self.data[self.datetime_column].min(),
            self.data[self.datetime_column].max(),
        )

    def get_ticker(self, ticker: str) -> pd.DataFrame:
        return (
            self.data[self.data[self.ticker_column] == ticker]
            .sort_values(self.datetime_column)
            .reset_index(drop=True)
        )