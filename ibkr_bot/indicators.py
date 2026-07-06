"""Indicator series for the Config-A bot (ATR stop width, HTF trend filter).

These implementations are copied verbatim from the authoritative
`oanda_bot/backtest.py` (`compute_atr_series` / `compute_ema_series`) — the
live bot's ATR and EMA must be computed exactly the way the backtest computes
them, or the validated edge no longer applies to live trades.
"""

from datetime import datetime
from typing import Optional

from ibkr_client import Candle


def compute_atr_series(candles: list[Candle], period: int = 14) -> dict[datetime, float]:
    """Wilder ATR per candle.time. Uses prior close for true range."""
    out: dict[datetime, float] = {}
    trs: list[float] = []
    prev_close: Optional[float] = None
    atr: Optional[float] = None
    for i, c in enumerate(candles):
        if prev_close is None:
            tr = c.high - c.low
        else:
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        if i + 1 == period:
            atr = sum(trs[:period]) / period
            out[c.time] = atr
        elif i + 1 > period:
            atr = (atr * (period - 1) + tr) / period
            out[c.time] = atr
        prev_close = c.close
    return out


def compute_ema_series(candles: list[Candle], period: int) -> dict[datetime, float]:
    """EMA of close per candle.time."""
    out: dict[datetime, float] = {}
    k = 2 / (period + 1)
    ema: Optional[float] = None
    for c in candles:
        ema = c.close if ema is None else (c.close * k + ema * (1 - k))
        out[c.time] = ema
    return out


def value_at_or_last(
    series: dict[datetime, float],
    at: datetime,
) -> tuple[Optional[float], bool]:
    """Look up a series value at an exact candle time, falling back to the
    latest available value.

    Returns (value, exact) where exact is False when the fallback was used.
    The fallback covers the race where the just-closed breakout candle is not
    yet in the fetched history — one bar of staleness, logged by the caller.
    """
    if at in series:
        return series[at], True
    if series:
        return series[max(series)], False
    return None, False
