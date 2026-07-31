from __future__ import annotations
from operator import ge

import pandas as pd

from ta.trend import (
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    SMAIndicator,
    MACD,
    ADXIndicator,
)

from core.columns import Columns
from src.data.market_frame import MarketFrame
from src.features.feature_generator import FeatureGeneratorInterface


class TrendFeatures(FeatureGeneratorInterface):

    def __init__(
        self,
        ema_periods: tuple[int, ...] = (10, 20, 50, 100, 200),
        sma_periods: tuple[int, ...] = (20, 50, 200),
    ):
        self._ema_periods = ema_periods
        self._sma_periods = sma_periods

    @property
    def name(self) -> str:
        return "Trend Features"

    def transform(
        self,
        frame: MarketFrame,
    ) -> None:

        df = frame.data

        generated = pd.DataFrame(index=df.index)

        for group in frame.iter_tickers():

            idx = group.index

            close = group[Columns.CLOSE]
            high = group[Columns.HIGH]
            low = group[Columns.LOW]

            # EMA
            for period in self._ema_periods:
                generated.loc[idx, f"ema_{period}"] = EMAIndicator(
                    close,
                    window=period,
                ).ema_indicator()

            # SMA
            for period in self._sma_periods:
                generated.loc[idx, f"sma_{period}"] = SMAIndicator(
                    close,
                    window=period,
                ).sma_indicator()

            # MACD
            macd = MACD(close)

            generated.loc[idx, "macd"] = macd.macd()
            generated.loc[idx, "macd_signal"] = macd.macd_signal()
            generated.loc[idx, "macd_hist"] = macd.macd_diff()

            # ADX
            adx = ADXIndicator(
                high=high,
                low=low,
                close=close,
            )

            generated.loc[idx, "adx"] = adx.adx()
            generated.loc[idx, "adx_pos"] = adx.adx_pos()
            generated.loc[idx, "adx_neg"] = adx.adx_neg()
            generated.loc[idx, "adx_spread"] = adx.adx_pos() - adx.adx_neg()

            # Aroon Indicator
            aroon = AroonIndicator(close, low)
            generated.loc[idx, "aroon_indicator"] = aroon.aroon_indicator()
            generated.loc[idx, "aroon_up"] = aroon.aroon_up()
            generated.loc[idx, "aroon_down"] = aroon.aroon_down()
            generated.loc[idx, "aroon_oscillator"] = aroon.aroon_up() - aroon.aroon_down()

            # Commodity Channel Index
            generated.loc[idx, "cci"] = CCIIndicator(high, low, close).cci()

            # Detrended Price Oscillator
            generated.loc[idx, "dpo"] = DPOIndicator(close).dpo()

            # Ichimoku Kinko Hyo
            ichimoku = IchimokuIndicator(high, low) # INFO: visual must stay False to not introduce future information

            generated.loc[idx, "ichimoku_conversion"] = ichimoku.ichimoku_conversion_line() # INFO: Tenkan-sen
            generated.loc[idx, "ichimoku_base"] = ichimoku.ichimoku_base_line()             # INFO: Kijun-sen
            generated.loc[idx, "ichimoku_a"] = ichimoku.ichimoku_a()
            generated.loc[idx, "ichimoku_b"] = ichimoku.ichimoku_b()
            generated.loc[idx, "ichimoku_tk_diff"] = ichimoku.ichimoku_conversion_line() - ichimoku.ichimoku_base_line()
            generated.loc[idx, "ichimoku_cloud_width"] = ichimoku.ichimoku_a() - ichimoku.ichimoku_b()
            generated.loc[idx, "ichimoku_price_to_base"] = clsoe - ichimoku.ichimoku_base_line()
            generated.loc[idx, "ichimoku_price_to_conversion"] = close - ichimoku.ichimoku_conversion_line()

            # Know Sure Thing Oscillator
            kst = KSTIndicator(close)

            generated.loc[idx, "kst_indicator"] = kst.kst()
            generated.loc[idx, "kst_diff"] = kst.kst_diff()
            generated.loc[idx, "kst_sig"] = kst.kst_sig()

        frame.add_features(generated)
