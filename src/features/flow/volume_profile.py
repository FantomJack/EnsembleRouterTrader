# TODO: 
# volume_profile_poc
# volume_profile_val
# volume_profile_vah
# volume_profile_width
# volume_profile_width_pct
# close_to_poc
# close_to_val
# close_to_vah
# poc_change
# poc_change_pct
# volume_profile_skewness
# volume_profile_kurtosis
# vp_jarque_bera
# vp_normality_pvalue


# WARNING: I must find out, what price bin size I should use.
# or does it matter? How much granularity do I actually need?
#

# TODO:
# Location
# vp_poc
# vp_vwap / volume-weighted mean price
# vp_median
# vp_q10
# vp_q25
# vp_q50
# vp_q75
# vp_q90

# TODO:
# Dispersion
# vp_variance
# vp_std
# vp_range
# vp_iqr
# vp_cv

# TODO: 
# Shape
# vp_skewness
# vp_kurtosis

# TODO: 
# Distribution / normality
# vp_normality_stat
# vp_normality_pvalue

# TODO:
# Value-area structure
# vp_val
# vp_vah
# vp_value_area_width
# vp_num_peaks
# vp_peak_separation
# vp_primary_peak_volume_ratio

# TODO: 
# Relative-to-price
# (close - vp_poc) / close
# (close - vp_median) / close
# (close - vp_mean) / close
# vp_std / close
# vp_iqr / close
# (vp_vah - vp_val) / close#


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import chi2

from src.core.columns import Columns
from src.data.market_frame import MarketFrame
from src.features.feature_generator import FeatureGeneratorInterface


@dataclass(slots=True)
class _VolumeProfile:
    prices: np.ndarray
    volumes: np.ndarray


class VolumeProfileFeatures(FeatureGeneratorInterface):
    """
    Builds daily volume-at-price profiles from intraday OHLCV data.

    The input `intraday_frame` should contain 30-minute OHLCV data.
    The output features are attached to the daily MarketFrame passed
    to transform().

    Volume inside each intraday candle is distributed across price
    bins according to the overlap between the candle's [Low, High]
    interval and each price bin.

    This is an approximation because OHLCV data does not contain
    trade-level prices.
    """

    def __init__(
        self,
        intraday_frame: MarketFrame,
        n_bins: int = 50,
        value_area_pct: float = 0.70,
        peak_prominence: float = 0.05,
        peak_distance: int = 2,
    ) -> None:
        super().__init__()

        if n_bins < 5:
            raise ValueError("n_bins must be at least 5.")

        if not 0 < value_area_pct < 1:
            raise ValueError(
                "value_area_pct must be between 0 and 1."
            )

        if peak_prominence < 0:
            raise ValueError(
                "peak_prominence must be non-negative."
            )

        if peak_distance < 1:
            raise ValueError(
                "peak_distance must be at least 1."
            )

        self._intraday_frame = intraday_frame
        self._n_bins = n_bins
        self._value_area_pct = value_area_pct
        self._peak_prominence = peak_prominence
        self._peak_distance = peak_distance

    @property
    def name(self) -> str:
        return "Volume Profile Features"

    def transform(self, frame: MarketFrame) -> None:
        """
        Build daily volume-profile features from intraday data
        and attach them to `frame`.
        """

        generated = pd.DataFrame(index=frame.data.index)

        intraday_profiles = self._build_profiles()

        for group in frame.iter_tickers():

            idx = group.index

            if not isinstance(idx, pd.DatetimeIndex):
                raise TypeError(
                    "MarketFrame index must be a DatetimeIndex "
                    "for VolumeProfileFeatures."
                )

            close = group[Columns.CLOSE]

            ticker = self._get_ticker_from_group(group)

            for timestamp in idx:
                day = timestamp.normalize()

                profile = intraday_profiles.get(
                    (ticker, day)
                )

                if profile is None:
                    continue

                features = self._calculate_features(
                    profile=profile,
                    close=float(close.loc[timestamp]),
                )

                for name, value in features.items():
                    generated.loc[timestamp, name] = value

        frame.add_features(generated)

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def _build_profiles(
        self,
    ) -> dict[tuple[object, pd.Timestamp], _VolumeProfile]:

        profiles: dict[
            tuple[object, pd.Timestamp],
            _VolumeProfile,
        ] = {}

        for group in self._intraday_frame.iter_tickers():

            if not isinstance(group.index, pd.DatetimeIndex):
                raise TypeError(
                    "Intraday MarketFrame index must be a "
                    "DatetimeIndex."
                )

            ticker = self._get_ticker_from_group(group)

            data = group.copy()
            data["_profile_day"] = data.index.normalize()

            for day, day_data in data.groupby(
                "_profile_day",
                sort=True,
            ):
                profile = self._build_daily_profile(day_data)

                if profile is not None:
                    profiles[(ticker, day)] = profile

        return profiles

    def _build_daily_profile(
        self,
        data: pd.DataFrame,
    ) -> _VolumeProfile | None:

        high = data[Columns.HIGH].astype(float)
        low = data[Columns.LOW].astype(float)
        volume = data[Columns.VOLUME].astype(float)

        valid = (
            high.notna()
            & low.notna()
            & volume.notna()
            & (volume >= 0)
            & (high >= low)
        )

        high = high[valid]
        low = low[valid]
        volume = volume[valid]

        if high.empty:
            return None

        profile_low = float(low.min())
        profile_high = float(high.max())

        if not np.isfinite(profile_low) or not np.isfinite(profile_high):
            return None

        if profile_high <= profile_low:
            # Entire day traded at one price.
            price = profile_low

            return _VolumeProfile(
                prices=np.array([price]),
                volumes=np.array([float(volume.sum())]),
            )

        bin_edges = np.linspace(
            profile_low,
            profile_high,
            self._n_bins + 1,
        )

        bin_width = bin_edges[1] - bin_edges[0]

        prices = (
            bin_edges[:-1]
            + bin_edges[1:]
        ) / 2.0

        profile_volume = np.zeros(
            self._n_bins,
            dtype=float,
        )

        for candle_low, candle_high, candle_volume in zip(
            low.to_numpy(),
            high.to_numpy(),
            volume.to_numpy(),
        ):
            if candle_volume <= 0:
                continue

            if candle_high <= candle_low:
                bin_index = np.searchsorted(
                    bin_edges,
                    candle_low,
                    side="right",
                ) - 1

                bin_index = int(
                    np.clip(
                        bin_index,
                        0,
                        self._n_bins - 1,
                    )
                )

                profile_volume[bin_index] += candle_volume
                continue

            candle_range = candle_high - candle_low

            overlap_low = np.maximum(
                bin_edges[:-1],
                candle_low,
            )

            overlap_high = np.minimum(
                bin_edges[1:],
                candle_high,
            )

            overlap = np.maximum(
                0.0,
                overlap_high - overlap_low,
            )

            weights = overlap / candle_range

            profile_volume += candle_volume * weights

        return _VolumeProfile(
            prices=prices,
            volumes=profile_volume,
        )

    # ------------------------------------------------------------------
    # Feature calculation
    # ------------------------------------------------------------------

    def _calculate_features(
        self,
        profile: _VolumeProfile,
        close: float,
    ) -> dict[str, float]:

        prices = profile.prices
        volumes = profile.volumes

        total_volume = volumes.sum()

        if total_volume <= 0:
            return {}

        weights = volumes / total_volume

        # --------------------------------------------------------------
        # Location
        # --------------------------------------------------------------

        mean_price = float(
            np.sum(weights * prices)
        )

        median_price = self._weighted_quantile(
            prices,
            weights,
            0.50,
        )

        q10 = self._weighted_quantile(
            prices,
            weights,
            0.10,
        )

        q25 = self._weighted_quantile(
            prices,
            weights,
            0.25,
        )

        q75 = self._weighted_quantile(
            prices,
            weights,
            0.75,
        )

        q90 = self._weighted_quantile(
            prices,
            weights,
            0.90,
        )

        poc_index = int(
            np.argmax(volumes)
        )

        poc = float(prices[poc_index])

        # --------------------------------------------------------------
        # Dispersion
        # --------------------------------------------------------------

        variance = float(
            np.sum(
                weights
                * (prices - mean_price) ** 2
            )
        )

        std = float(
            np.sqrt(max(variance, 0.0))
        )

        iqr = float(q75 - q25)

        price_range = float(
            prices.max() - prices.min()
        )

        cv = (
            std / abs(mean_price)
            if mean_price != 0
            else np.nan
        )

        # --------------------------------------------------------------
        # Shape
        # --------------------------------------------------------------

        if std > 0:
            standardized = (
                prices - mean_price
            ) / std

            skewness = float(
                np.sum(
                    weights
                    * standardized ** 3
                )
            )

            kurtosis_excess = float(
                np.sum(
                    weights
                    * standardized ** 4
                )
                - 3.0
            )
        else:
            skewness = 0.0
            kurtosis_excess = 0.0

        # --------------------------------------------------------------
        # Jarque-Bera
        # --------------------------------------------------------------

        effective_n = self._effective_sample_size(
            weights
        )

        if effective_n > 1:
            jarque_bera = (
                effective_n / 6.0
            ) * (
                skewness ** 2
                + (kurtosis_excess ** 2) / 4.0
            )

            normality_pvalue = float(
                chi2.sf(
                    jarque_bera,
                    df=2,
                )
            )
        else:
            jarque_bera = np.nan
            normality_pvalue = np.nan

        # --------------------------------------------------------------
        # Value area
        # --------------------------------------------------------------

        val, vah = self._calculate_value_area(
            prices,
            volumes,
            poc_index,
        )

        value_area_width = vah - val

        # --------------------------------------------------------------
        # Peaks
        # --------------------------------------------------------------

        peak_features = self._calculate_peak_features(
            prices,
            volumes,
        )

        # --------------------------------------------------------------
        # Relative-to-price
        # --------------------------------------------------------------

        features = {
            # Location
            "vp_poc": poc,
            "vp_vwap": mean_price,
            "vp_median": median_price,
            "vp_q10": q10,
            "vp_q25": q25,
            "vp_q50": median_price,
            "vp_q75": q75,
            "vp_q90": q90,

            # Dispersion
            "vp_variance": variance,
            "vp_std": std,
            "vp_range": price_range,
            "vp_iqr": iqr,
            "vp_cv": cv,

            # Shape
            "vp_skewness": skewness,
            "vp_kurtosis": kurtosis_excess,

            # Normality
            "vp_jarque_bera": jarque_bera,
            "vp_normality_pvalue": normality_pvalue,

            # Value area
            "vp_val": val,
            "vp_vah": vah,
            "vp_value_area_width": value_area_width,

            # Relative to current price
            "close_to_poc": self._relative_distance(
                close,
                poc,
            ),

            "close_to_val": self._relative_distance(
                close,
                val,
            ),

            "close_to_vah": self._relative_distance(
                close,
                vah,
            ),

            "close_to_vp_median": self._relative_distance(
                close,
                median_price,
            ),

            "close_to_vp_mean": self._relative_distance(
                close,
                mean_price,
            ),

            "vp_std_pct": self._relative_to_price(
                std,
                close,
            ),

            "vp_iqr_pct": self._relative_to_price(
                iqr,
                close,
            ),

            "vp_value_area_width_pct": self._relative_to_price(
                value_area_width,
                close,
            ),
        }

        features.update(peak_features)

        return features

    # ------------------------------------------------------------------
    # Value area
    # ------------------------------------------------------------------

    def _calculate_value_area(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        poc_index: int,
    ) -> tuple[float, float]:

        target_volume = (
            volumes.sum()
            * self._value_area_pct
        )

        included = np.zeros(
            len(volumes),
            dtype=bool,
        )

        included[poc_index] = True

        accumulated = volumes[poc_index]

        left = poc_index - 1
        right = poc_index + 1

        while accumulated < target_volume:

            left_volume = (
                volumes[left]
                if left >= 0
                else -np.inf
            )

            right_volume = (
                volumes[right]
                if right < len(volumes)
                else -np.inf
            )

            if left_volume == -np.inf and right_volume == -np.inf:
                break

            if right_volume >= left_volume:
                included[right] = True
                accumulated += volumes[right]
                right += 1
            else:
                included[left] = True
                accumulated += volumes[left]
                left -= 1

        included_prices = prices[included]

        return (
            float(included_prices.min()),
            float(included_prices.max()),
        )

    # ------------------------------------------------------------------
    # Peaks
    # ------------------------------------------------------------------

    def _calculate_peak_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> dict[str, float]:

        if len(volumes) < 3:
            return {
                "vp_num_peaks": 1.0,
                "vp_peak_separation": 0.0,
                "vp_primary_peak_volume_ratio": 1.0,
            }

        max_volume = volumes.max()

        if max_volume <= 0:
            return {
                "vp_num_peaks": 0.0,
                "vp_peak_separation": np.nan,
                "vp_primary_peak_volume_ratio": np.nan,
            }

        peaks, properties = find_peaks(
            volumes,
            prominence=max_volume * self._peak_prominence,
            distance=self._peak_distance,
        )

        if len(peaks) == 0:
            primary_index = int(
                np.argmax(volumes)
            )

            return {
                "vp_num_peaks": 1.0,
                "vp_peak_separation": 0.0,
                "vp_primary_peak_volume_ratio": 1.0,
            }

        prominences = properties["prominences"]

        primary_order = np.argsort(
            volumes[peaks]
        )[::-1]

        primary_index = peaks[
            primary_order[0]
        ]

        primary_volume = volumes[
            primary_index
        ]

        if len(peaks) >= 2:
            ordered_peaks = peaks[
                np.argsort(volumes[peaks])[::-1]
            ]

            second_index = ordered_peaks[1]

            separation = abs(
                prices[primary_index]
                - prices[second_index]
            )
        else:
            separation = 0.0

        return {
            "vp_num_peaks": float(len(peaks)),
            "vp_peak_separation": float(separation),
            "vp_primary_peak_volume_ratio": float(
                primary_volume / volumes.sum()
            ),
        }

    # ------------------------------------------------------------------
    # Weighted statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:

        order = np.argsort(values)

        values = values[order]
        weights = weights[order]

        cumulative = np.cumsum(weights)

        index = np.searchsorted(
            cumulative,
            quantile,
            side="left",
        )

        index = min(
            index,
            len(values) - 1,
        )

        return float(values[index])

    @staticmethod
    def _effective_sample_size(
        weights: np.ndarray,
    ) -> float:

        denominator = np.sum(
            weights ** 2
        )

        if denominator <= 0:
            return 0.0

        return float(
            1.0 / denominator
        )

    @staticmethod
    def _relative_distance(
        price: float,
        reference: float,
    ) -> float:

        if price == 0:
            return np.nan

        return float(
            (price - reference) / price
        )

    @staticmethod
    def _relative_to_price(
        value: float,
        price: float,
    ) -> float:

        if price == 0:
            return np.nan

        return float(
            value / price
        )

    # ------------------------------------------------------------------
    # Ticker handling
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ticker_from_group(
        group: pd.DataFrame,
    ) -> object:

        if Columns.TICKER in group.columns:
            return group[Columns.TICKER].iloc[0]

        # If iter_tickers() already removes the ticker column,
        # this should be replaced by the actual ticker identifier
        # exposed by MarketFrame.
        raise ValueError(
            "Ticker information is required for "
            "VolumeProfileFeatures. "
            "Add the ticker to the grouped data or expose it "
            "through MarketFrame.iter_tickers()."
        )
