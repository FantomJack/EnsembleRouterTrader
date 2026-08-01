from __future__ import annotations

from numpy import sign

from core.columns import Columns
from src.features.feature_generator import FeatureGeneratorInterface
from src.data.market_frame import MarketFrame

from ta.volume import (
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    VolumeWeightedAveragePrice
)

class VolumeFeatures(FeatureGeneratorInterface):

    def __init__(
            self,
            volume_window,
            roc_window
    ) -> None:
        self._volume_window = volume_window
        self._roc_window = roc_window

    @property
    def name(self) -> str:
        return "Volume Features"

    def transform(self, frame: MarketFrame) -> None:

        generated = frame.data

        for group in frame.iter_tickers():

            idx = group.index

            high = group[Columns.HIGH]
            low = group[Columns.LOW]
            close = group[Columns.CLOSE]
            volume = group[Columns.VOLUME]

            # Price Volume Trend


            # Ease of movement
            eom = EaseOfMovementIndicator(high, low, volume)
            generated.loc[idx, "ease_of_movement"] = eom.ease_of_movement()
            generated.loc[idx, "sma_eom"] = eom.sma_ease_of_movement()

            # Force Index Indicator
            generated.loc[idx, "force_index"] = ForceIndexIndicator(close, volume).force_index()

            # Money Flow Index
            generated.loc[idx, "mfi"] = MFIIndicator(high, low, close, volume).money_flow_index()

            # Money Flow Multiplier and Volume
            mfm = (2*close - low - high) / (high - low)
            generated.loc[idx, "mfm"] = mfm
            generated.loc[idx, "mfv"] = mfm * volume

            # Signed Volume
            generated.loc[idx, "signed_volume"] = sign(close.diff()) * volume

            # Buy/Sell Pressure Proxy
            generated.loc[idx, "bs_pressure"] = (close - open) / (high - low)
 
            # Volume Weighted Average Price
            vwap = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
            generated.loc[idx, "vwap"] = vwap
            generated.loc[idx, "price_to_vwap"] = (close - vwap) / vwap
