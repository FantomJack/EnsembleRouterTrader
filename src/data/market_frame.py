from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from src.core.columns import Columns

@dataclass(slots=True)
class MarketFrame:
    """
    Represents OHLCV market data from CSV file.

    The class intentionally does not compute technical indicators,
    labels, or machine learning features. Those belong to later stages
    of the data pipeline.
    """
    
    data: pd.DataFrame

    feature_columns: list[str] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


    def __post_init__(self):

        if self.data.empty:
            raise EmptyDatasetException()

        self.data = self.data.sort_values(
            [Columns.TICKER, Columns.DATE]
        )

        self.data.reset_index(drop=True, inplace=True)

        if self.data.duplicated(
                [Columns.TICKER, Columns.DATE]
        ).any():
            raise DuplicateRowsException()


    @property
    def tickers(self) -> list[str]:
        return sorted(self.data[Columns.TICKER].unique())

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def start_date(self):
        return self.data[Columns.DATE].min()

    @property
    def end_date(self):
        return self.data[Columns.DATE].max()

    @property
    def date_range(self):
        return (
            self.start_date,
            self.end_date,
        )

    def get_ticker(self, ticker: str) -> pd.DataFrame:
        return (
            self.data[
                self.data[Columns.TICKER] == ticker
            ]
            .sort_values(Columns.DATE)
            .reset_index(drop=True)
        )

    def add_features(
            self,
            features: pd.DataFrame,
    ) -> None:

        duplicate = set(features.columns) & set(self.data.columns)

        if duplicate:
            raise ValueError(
                f"Features already exist: {duplicate}"
            )

        if len(features) != len(self.data):
            raise ValueError(
                "Features must have the same number of rows "
                "as the market data."
            )

        if not features.index.equals(self.data.index):
            raise ValueError(
                "Features must have the same index as the market data."
            )

        self.data = pd.concat(
            [self.data, features],
            axis=1,
        )

        self.feature_columns.extend(features.columns)

    def add_targets(
        self,
        targets: pd.DataFrame,
    ) -> None:
        duplicate = set(targets.columns) & set(self.data.columns)

        if duplicate:
            raise ValueError(
                f"Targets already exist: {duplicate}"
            )

        if len(targets) != len(self.data):
            raise ValueError(
                "Targets must have the same number of rows "
                "as the market data."
            )

        if not targets.index.equals(self.data.index):
            raise ValueError(
                "Targets must have the same index as the market data."
            )

        self.data = pd.concat(
            [self.data, targets],
            axis=1,
        )

        self.target_columns.extend(targets.columns)

    def add_target(self, name: str) -> None:
        if name not in self.data.columns:
            raise ValueError(
                f"Target column does not exist: {name}"
            )

        if name not in self.target_columns:
            self.target_columns.append(name)

    def iter_tickers(self):
        for _, group in self.data.groupby(Columns.TICKER, sort=False):
            yield group
