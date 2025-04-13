from enum import Enum


class TimeInterval(Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "day"
    WEEKLY = "week"
    MONTHLY = "month"
    QUARTERLY = "quarter"
    YEARLY = "year"


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class Moneyness(Enum):
    ITM = "ITM"  # In The Money
    ATM = "ATM"  # At The Money
    OTM = "OTM"  # Out of The Money
