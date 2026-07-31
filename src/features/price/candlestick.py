from __future__ import annotations
from operator import ge

import pandas as pd

from src.data.market_frame import MarketFrame
from src.features.feature_generator import FeatureGeneratorInterface
from src.core.columns import Columns

class CandlestickFeatures(FeatureGeneratorInterface):
    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "Candlestick Features"

    def transform(self, frame: MarketFrame) -> None:

        generated = pd.DataFrame(index=frame.data.index)

        for group in frame.iter_tickers():

            idx = group.index

            open_ = group[Columns.OPEN]
            close = group[Columns.CLOSE]
            high = group[Columns.HIGH]
            low = group[Columns.LOW]
            volume = group[Columns.VOLUME]

            body = close - open_
            range_ = (high - low).mask((high - low) == 0)

            generated.loc[idx, "body"] = body
            generated.loc[idx, "body_abs"] = body.abs()
            generated.loc[idx, "body_pct"] = body / open_

            generated.loc[idx, "range"] = range_
            generated.loc[idx, "range_pct"] = range_ / close

            oc_min = pd.concat([open_, close], axis=1).min(axis=1)
            oc_max = pd.concat([open_, close], axis=1).max(axis=1)

            generated.loc[idx, "lower_shadow"] = oc_min - low
            generated.loc[idx, "lower_shadow_pct"] = (oc_min - low) / range_

            generated.loc[idx, "upper_shadow"] = high - oc_max
            generated.loc[idx, "upper_shadow_pct"] = (high - oc_max) / range_

            generated.loc[idx, "open_position"] = (open_ - low) / range_
            generated.loc[idx, "close_position"] = (close - low) / range_
            generated.loc[idx, "body_position"] = (close - low) / range_
            generated.loc[idx, "body_ratio"] = body.abs() / range_

            generated.loc[idx, "bullish"] = (close > open_).astype(int)
            generated.loc[idx, "bearish"] = (close < open_).astype(int)

            prev_close = close.shift(1)

            generated.loc[idx, "gap"] = open_ - prev_close
            generated.loc[idx, "gap_pct"] = (open_ - prev_close) / prev_close
            generated.loc[idx, "overnight_return"] = (open_ - prev_close) / prev_close

            generated.loc[idx, "intraday_return"] = body / open_

            rolling_high = high.rolling(20).max()
            rolling_low = low.rolling(20).min()

            generated.loc[idx, "close_in_range_20"] = (
                (close - rolling_low) / (rolling_high - rolling_low)
            )
