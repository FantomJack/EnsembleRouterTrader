from __future__ import annotations
from operator import ge

import pandas as pd

from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex
)

from core.columns import Columns
from src.data.market_frame import MarketFrame
from src.features.feature_generator import FeatureGeneratorInterface



# TODO: maybe 
# Historical Volatility
# Rolling Std
# Parkinson Volatility
# Donchian Width

class VolatilityFeatures(FeatureGeneratorInterface):

    def __init__(
        self,
        historical_volatility_window: int = 20
    ):
        self._historical_volatility_window = historical_volatility_window

    @property
    def name(self) -> str:
        return "Volatility Features"

    def transform(
        self,
        frame : MarketFrame
    ) -> None :
        
        generated = pd.DataFrame(index = frame.data.index)

        for group in frame.iter_tickers():

            idx = group.index

            high = group[frame.high_column]
            low = group[frame.low_column]
            close = group[Columns.CLOSE]
            # ATR
            generated.loc[idx, "ATR"] = AverageTrueRange(high, low, close).average_true_range()

            # Bollinger Bands
            b_bands = BollingerBands(close)

            generated.loc[idx, "bb_upper"] = b_bands.bollinger_hband()
            generated.loc[idx, "bb_middle"] = b_bands.bollinger_mavg()
            generated.loc[idx, "bb_lower"] = b_bands.bollinger_lband()
            generated.loc[idx, "bb_percent"] = b_bands.bollinger_pband()
            generated.loc[idx, "bb_width"] = b_bands.bollinger_wband()

            # Donchian Channel
            dc = DonchianChannel(high, low, close)

            generated.loc[idx, "donchian_upper"] = dc.donchian_channel_hband()
            generated.loc[idx, "donchian_middle"] = dc.donchian_channel_mband()
            generated.loc[idx, "donchian_lower"] = dc.donchian_channel_lband()
            generated.loc[idx, "donchian_width"] = dc.donchian_channel_wband()
            generated.loc[idx, "donchian_percent"] = dc.donchian_channel_pband()

            # Keltner Channel
            kc = KeltnerChannel(high, low, close)

            generated.loc[idx, "keltner_upper"] = kc.keltner_channel_hband()
            generated.loc[idx, "keltner_middle"] = kc.keltner_channel_mband()
            generated.loc[idx, "keltner_lower"] = kc.keltner_channel_lband()
            generated.loc[idx, "keltner_width"] = kc.keltner_channel_wband()
            generated.loc[idx, "keltner_percent"] = kc.keltner_channel_pband()

            # Ulcer Index
            generated.loc[idx, "ulcer_index"] = UlcerIndex(close).ulcer_index()

            # Historical Volatility
            generated.loc[idx, "historical_volatility"] = (
                close.pct_change().rolling(self.historical_volatility_window).std() * (252 ** 0.5)
            )

        frame.add_features(generated)
