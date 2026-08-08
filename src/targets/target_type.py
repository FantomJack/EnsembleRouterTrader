from enum import Enum


class TargetType(str, Enum):
    RETURN = "return"
    LOG_RETURN = "log_return"
    DIRECTION = "direction"
    # TODO: Tieto treba este dorobit do tarbet_builder.py
    VOLATILITY = "volatility"
    RANK = 'rank',
    REGIME = 'regime'
