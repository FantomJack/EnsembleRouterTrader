import numpy as np
import pandas as pd

from core.columns import Columns
from src.data.market_frame import MarketFrame
from src.targets.target_type import TargetType

class TargetBuilder:

    def __init__(
        self,
        window_sizes : tuple[int, ...] = (1, 5, 10, 20),
        target_types=[
                TargetType.RETURN,
                TargetType.DIRECTION
        ]
    ) -> None:
        self._window_sizes = window_sizes
        self._target_types = target_types

    def transform(
        self,
        frame: MarketFrame
    ) -> None:

        generated = pd.DataFrame(index = frame.data.index)

        for group in frame.iter_tickers():
            idx = group.index
            close = group[Columns.CLOSE]

            for window in self._window_sizes:
                future_close = close.diff(window)

                if TargetType.RETURN in self._target_types:
                    generated.loc[idx, f"future_return_{window}"] = (future_close - close) / close
                
                if TargetType.LOG_RETURN in self._target_types:
                    generated.loc[idx, f"future_log_return_{window}"] = np.log(future_close / close)
                
                if TargetType.DIRECTION in self._target_types:
                    generated.loc[idx, f"future_direction_{window}"] = (future_close > close).astype(int)
                    
                    threshold = 0.01

                    returns = (future_close - close) / close

                    target = np.select(
                        [returns > threshold, returns < -threshold],
                        [ 1, -1], default=0,
                    )

                    generated.loc[idx, f"future_direction_wt_{window}"] = target
                
                # TODO: generated.loc[idx, f"future_relative_return_{window}"] = 
                # future_up_1pct, future_down_1pct
        
        frame.add_targets(generated)
