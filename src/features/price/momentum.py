from __future__ import annotations

import pandas as pd

from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
    ROCIndicator,
    AwesomeOscillatorIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    KAMAIndicator,
)

from src.core.columns import Columns
from src.data.market_frame import MarketFrame
from src.features.feature_generator import FeatureGeneratorInterface


class MomentumFeatures(FeatureGeneratorInterface):

    # TODO: finish configurability of the window size
    def __init__(
            self,
            rsi_window: int = 14,
            roc_window: int = 10,
            stochastic_window: int = 14,
    ):
        self._rsi_window = rsi_window
        self._roc_window = roc_window
        self._stochastic_window = stochastic_window

    @property
    def name(self) -> str:
        return "Momentum Features"

    def transform(self, frame: MarketFrame) -> None:

        generated = pd.DataFrame(index=frame.data.index)


        # FIX: This might not work if not used properly
        for group in frame.iter_tickers():

            idx = group.index

            close = group[Columns.CLOSE]
            high = group[Columns.HIGH]
            low = group[Columns.LOW]
            volume = group[Columns.VOLUME]

            # RSI
            generated.loc[idx, "rsi_14"] = RSIIndicator(close, 14).rsi()

            # TSI
            generated.loc[idx, "tsi"] = RSIIndicator(close).tsi()

            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=high,
                low=low,
                close=close,
            )

            generated.loc[idx, "stoch_k"] = stoch.stoch()
            generated.loc[idx, "stoch_d"] = stoch.stoch_signal()

            # Williams %R
            generated.loc[idx, "williams_r"] = WilliamsRIndicator(
                high,
                low,
                close,
            ).williams_r()

            # ROC
            generated.loc[idx, "roc_10"] = ROCIndicator(close).roc()

            # Awesome Oscillator
            generated.loc[idx, "awesome_oscillator"] = (
                AwesomeOscillatorIndicator(
                    high,
                    low,
                ).awesome_oscillator()
            )

            # Ultimate OScillator
            generated.loc[idx, "ultimate_oscilator"] = UltiimateOscilaltor(high, low, close).ultimate_oscilator()

            # PPO
            ppo = PercentagePriceOscillator(close)

            generated.loc[idx, "ppo"] = ppo.ppo()
            generated.loc[idx, "ppo_signal"] = ppo.ppo_signal()
            generated.loc[idx, "ppo_hist"] = ppo.ppo_hist()

            # PVO
            pvo = PercentageVolumeOscillator(volume)

            generated.loc[idx, "pvo"] = pvo.pvo()
            generated.loc[idx, "pvo_signal"] = pvo.pvo_signal()
            generated.loc[idx, "pvo_hist"] = pvo.pvo_hist()

            # KAMA
            generated.loc[idx, "kama"] = KAMAIndicator(close).kama()

        frame.add_features(generated)
