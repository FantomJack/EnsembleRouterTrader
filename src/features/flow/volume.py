from __future__ import annotations

import pandas as pd
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
)

from core.columns import Columns
from src.features.feature_generator import FeatureGeneratorInterface
from src.data.market_frame import MarketFrame

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

        generated = pd.DataFrame(index = frame.data.index)

        for group in frame.iter_tickers():

            idx = group.index

            high = group[Columns.HIGH]
            low = group[Columns.LOW]
            close = group[Columns.CLOSE]
            volume = group[Columns.VOLUME]

            # Accumulation / Distribution Index
            generated.loc[idx, "adi"] = AccDistIndexIndicator(high, low, close, volume).acc_dist_index()

            # Chaikin Money Flow
            generated.loc[idx, "chaikin"] = ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow()

            # Negative Volume Index
            generated.loc[idx, "negative_volume"] = NegativeVolumeIndexIndicator(close, volume).negative_volume_index()

            # On-balance volume
            obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            generated.loc[idx, "obv"] = obv
            generated.loc[idx, "obv_diff"] = obv.diff()

            # Volume-price trend
            generated.loc[idx, "vpt"] = VolumePriceTrendIndicator(close, volume).volume_price_trend()

            # Relative Volume
            volume_ma = volume.rolling(self._volume_window).mean()

            generated.loc[idx, "relative_volume"] = volume / volume_ma

            # Volume EMA
            generated.loc[idx, "volume_ema"] = volume.ewm(span=self._volume_window, adjust=False).mean()

            # Volume ROC
            generated.loc[idx, "volume_roc"] = volume.pct_change(self._roc_window)

            # Volume Z-score
            volume_std = volume.rolling(self._volume_window).std()

            generated.loc[idx, "volume_zscore"] = (volume - volume_ma) / volume_std

            # Dollar Volume
            dollar_volume = close * volume

            generated.loc[idx, "dollar_volume"] = dollar_volume

            generated.loc[idx, "relative_dollar_volume"] = (
                dollar_volume / dollar_volume.rolling(self._volume_window).mean()
            )

            # Volume percentile
            generated.loc[idx, "volume_percentile_252"] = volume.rolling(252).rank()
            generated.loc[idx, f"volume_percentile_{self._volume_window}"] = volume.rolling(self._volume_window).rank()

        frame.add_features(generated)
