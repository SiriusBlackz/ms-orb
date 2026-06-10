#!/usr/bin/env python3
"""MS-ORB Volume & ADX Filter Optimization Backtest.

Fetches historical M5 candle data from OANDA, simulates the MS-ORB strategy
across a grid of volume-ratio and ADX thresholds, and reports which filter
combination maximizes expectancy.

Usage:
    python backtest.py [--months 12] [--instruments XAU_USD,NAS100_USD]
                       [--sessions TOKYO,LONDON,NY] [--rr-target 5.0]
                       [--vol-filter None,1.0,1.25,1.5,2.0,2.5]
                       [--adx-filter None,15,20,25,30,35]
                       [--no-cache] [--verbose]
"""

import argparse
import itertools
import logging
import os
import pickle
import sys
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import pytz

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INSTRUMENT_CONFIG,
    SESSIONS,
    TIMEZONE,
    SessionConfig,
    SessionName,
    TradingConfig,
)
from oanda_client import Candle, OANDAClient
from strategy import (
    BreakoutDetector,
    BreakoutSignal,
    Direction,
    OpeningRange,
    RangeRecorder,
)

logger = logging.getLogger(__name__)

NY_TZ = pytz.timezone(TIMEZONE)

# Volume and ADX filter thresholds to test
# None = no filter; float = minimum ratio/value required
DEFAULT_VOL_THRESHOLDS: list[Optional[float]] = [None, 1.0, 1.25, 1.5, 2.0, 2.5]
DEFAULT_ADX_THRESHOLDS: list[Optional[float]] = [None, 15, 20, 25, 30, 35]

CACHE_DIR = Path(__file__).parent / "cache"


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    instrument: str
    session: SessionName
    trade_date: date
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str       # "stop", "target", "time"
    rr_achieved: float
    entry_time: datetime
    exit_time: datetime
    vol_threshold: Optional[float] = None
    adx_threshold: Optional[float] = None
    breakout_volume_ratio: Optional[float] = None
    breakout_adx: Optional[float] = None


@dataclass
class SimulationMetrics:
    """Aggregated metrics for one filter combination."""
    vol_threshold: Optional[float]
    adx_threshold: Optional[float]
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.rr_achieved > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.rr_achieved <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def avg_rr(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.rr_achieved for t in self.trades) / self.total_trades

    @property
    def total_r(self) -> float:
        return sum(t.rr_achieved for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.rr_achieved for t in self.trades if t.rr_achieved > 0)
        gross_loss = abs(sum(t.rr_achieved for t in self.trades if t.rr_achieved <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown_r(self) -> float:
        """Maximum drawdown measured in R."""
        if not self.trades:
            return 0.0
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        for t in self.trades:
            equity += t.rr_achieved
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def max_consecutive_losses(self) -> int:
        if not self.trades:
            return 0
        max_streak = 0
        streak = 0
        for t in self.trades:
            if t.rr_achieved <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def exit_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────
#  Volume & ADX indicators
# ─────────────────────────────────────────────────────────────


def compute_volume_ratio(
    breakout_candle: Candle,
    prior_candles: list[Candle],
    lookback: int = 20,
) -> Optional[float]:
    """Compare breakout candle volume to rolling average of prior candles.

    Args:
        breakout_candle: The candle that triggered the breakout signal.
        prior_candles: Session-local candles before the breakout (range + post-range so far).
        lookback: Number of prior candles to average over.

    Returns:
        Volume ratio (candle.volume / avg_volume), or None if insufficient data.
    """
    recent = prior_candles[-lookback:] if len(prior_candles) >= lookback else prior_candles
    if not recent:
        return None
    avg_volume = sum(c.volume for c in recent) / len(recent)
    if avg_volume <= 0:
        return None
    return breakout_candle.volume / avg_volume


def compute_adx_series(candles: list[Candle], period: int = 14) -> dict[datetime, float]:
    """Compute Wilder's ADX for a full candle series.

    Pure Python implementation — no external dependencies.
    Requires 2*period + 1 = 29 candles to warm up before producing values.

    Args:
        candles: Complete candle list sorted by time.
        period: ADX smoothing period (default 14).

    Returns:
        Dict mapping candle.time -> ADX value for each candle where ADX is available.
    """
    if len(candles) < 2:
        return {}

    # Step 1: Compute True Range, +DM, -DM for each candle pair
    tr_list: list[float] = []
    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []

    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        up_move = high - candles[i - 1].high
        down_move = candles[i - 1].low - low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period:
        return {}

    # Step 2: Wilder's smoothing for ATR, +DI, -DI
    # First value: simple sum of first `period` values
    atr = sum(tr_list[:period])
    plus_dm_smooth = sum(plus_dm_list[:period])
    minus_dm_smooth = sum(minus_dm_list[:period])

    dx_list: list[float] = []
    adx_series: dict[datetime, float] = {}

    for i in range(period, len(tr_list)):
        # Wilder's smoothing: prev - (prev / period) + current
        atr = atr - (atr / period) + tr_list[i]
        plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm_list[i]
        minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm_list[i]

        plus_di = (plus_dm_smooth / atr) * 100 if atr > 0 else 0
        minus_di = (minus_dm_smooth / atr) * 100 if atr > 0 else 0

        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0
        dx_list.append(dx)

        # ADX = Wilder's smooth of DX, needs `period` DX values to start
        if len(dx_list) == period:
            adx = sum(dx_list) / period
            # candle index: i+1 in the original candles list (offset by 1 for the TR calc)
            candle_idx = i + 1
            if candle_idx < len(candles):
                adx_series[candles[candle_idx].time] = adx
        elif len(dx_list) > period:
            adx = (adx * (period - 1) + dx) / period
            candle_idx = i + 1
            if candle_idx < len(candles):
                adx_series[candles[candle_idx].time] = adx

    return adx_series


# ─────────────────────────────────────────────────────────────
#  Data fetching (paginated)
# ─────────────────────────────────────────────────────────────


def fetch_candles_range(
    client: OANDAClient,
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    granularity: str = "M5",
    use_cache: bool = True,
) -> list[Candle]:
    """Fetch M5 candles for a date range, with pickle caching.

    Paginates using 'from' + count=5000 per request.
    """
    # Check cache
    cache_key = f"{instrument}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_{granularity}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached data: {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(f"Fetching {instrument} {granularity} candles: {start_dt.date()} to {end_dt.date()}")

    all_candles: list[Candle] = []
    current_from = start_dt

    import oandapyV20.endpoints.instruments as instruments_ep

    while current_from < end_dt:
        params = {
            "granularity": granularity,
            "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": 5000,
        }

        try:
            r = instruments_ep.InstrumentsCandles(instrument, params=params)
            client.api.request(r)
        except Exception as e:
            logger.error(f"API request failed: {e}")
            break

        raw_candles = r.response.get("candles", [])
        if not raw_candles:
            break

        for c in raw_candles:
            mid = c["mid"]
            candle_time = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))

            # Stop if we've passed end_dt
            if candle_time >= end_dt:
                break

            all_candles.append(Candle(
                time=candle_time,
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=int(c["volume"]),
                complete=c["complete"],
            ))

        # If fewer candles than requested, we've reached end of available data
        batch_size = len(raw_candles)

        # Advance cursor to last candle time + 1 second
        last_time = datetime.fromisoformat(raw_candles[-1]["time"].replace("Z", "+00:00"))
        if last_time >= end_dt:
            break
        current_from = last_time + timedelta(seconds=1)

        logger.info(f"  ... fetched {len(all_candles)} candles so far (batch: {batch_size})")

        if batch_size < 5000:
            break

        # Respect rate limits
        time_mod.sleep(0.5)

    logger.info(f"Total candles fetched for {instrument}: {len(all_candles)}")

    # Save to cache
    if use_cache and all_candles:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(all_candles, f)
        logger.info(f"Cached to {cache_file.name}")

    return all_candles


# ─────────────────────────────────────────────────────────────
#  Session day grouping
# ─────────────────────────────────────────────────────────────


def candle_ny_time(candle: Candle) -> datetime:
    """Convert a candle's UTC time to NY timezone."""
    utc_time = candle.time
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    return utc_time.astimezone(NY_TZ)


def assign_session_date(ny_time: datetime, session_cfg: SessionConfig) -> date:
    """Assign a NY-time candle to the correct session date.

    For Tokyo (18:00-02:54 NY), candles from 18:00-23:59 get today's date,
    candles from 00:00-02:54 get *yesterday's* date (the session started
    the prior calendar day).

    For London and NY, the session date is the calendar date of the candle.
    """
    t = ny_time.time()

    if session_cfg.name == SessionName.TOKYO:
        # Tokyo session spans midnight: range_start=18:00, session_close=02:54
        # Evening portion (18:00-23:59): session date = calendar date
        # Morning portion (00:00-02:54): session date = calendar date - 1
        if t >= session_cfg.range_start:
            return ny_time.date()
        elif t <= session_cfg.session_close:
            return (ny_time - timedelta(days=1)).date()
        else:
            return ny_time.date()  # outside session window
    else:
        return ny_time.date()


def is_in_session_window(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    """Check if a NY-timezone time falls within a session's full window."""
    t = ny_time.time()

    if session_cfg.name == SessionName.TOKYO:
        # Spans midnight: 18:00 -> 02:54 next day
        return t >= session_cfg.range_start or t <= session_cfg.session_close
    else:
        return session_cfg.range_start <= t <= session_cfg.session_close


def is_in_range_period(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    """Check if time is in the 30-min range period."""
    t = ny_time.time()
    return session_cfg.range_start <= t < session_cfg.range_end


def is_in_entry_window(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    """Check if time is in the entry window (range_end to entry_end)."""
    t = ny_time.time()

    if session_cfg.name == SessionName.TOKYO:
        # entry window: 18:30 to 21:30 (same evening, no midnight issue)
        return session_cfg.range_end <= t < session_cfg.entry_end
    else:
        return session_cfg.range_end <= t < session_cfg.entry_end


def is_past_session_close(ny_time: datetime, session_cfg: SessionConfig) -> bool:
    """Check if time is at or past session close."""
    t = ny_time.time()

    if session_cfg.name == SessionName.TOKYO:
        # Close is 02:54 next day. Past close if: morning and t >= close time
        # But also need to handle: in the morning portion, past 02:54
        if t <= session_cfg.session_close:
            return False  # still within session (before close)
        elif t < session_cfg.range_start:
            return True  # between 02:55 and 17:59 -> past close
        else:
            return False  # 18:00+ = new session evening
    else:
        return t >= session_cfg.session_close


def group_candles_by_session_day(
    candles: list[Candle],
    sessions: list[SessionConfig],
) -> dict[tuple[SessionName, date], list[Candle]]:
    """Group candles into (session, date) buckets.

    Returns dict mapping (SessionName, date) -> sorted list of candles
    belonging to that session day.
    """
    groups: dict[tuple[SessionName, date], list[Candle]] = {}

    for candle in candles:
        if not candle.complete:
            continue

        ny_time = candle_ny_time(candle)

        for session_cfg in sessions:
            if is_in_session_window(ny_time, session_cfg):
                session_date = assign_session_date(ny_time, session_cfg)
                key = (session_cfg.name, session_date)
                if key not in groups:
                    groups[key] = []
                groups[key].append(candle)

    # Sort each group by time
    for key in groups:
        groups[key].sort(key=lambda c: c.time)

    return groups


# ─────────────────────────────────────────────────────────────
#  Trade simulation
# ─────────────────────────────────────────────────────────────


def simulate_trade(
    signal: BreakoutSignal,
    candles: list[Candle],
    session_cfg: SessionConfig,
) -> Optional[TradeResult]:
    """Simulate a single trade from entry through exit (fixed SL, no BE).

    Walks candle-by-candle from the candle after entry.

    Per candle, checks in order:
    1. Session close -> time exit
    2. SL hit
    3. TP hit
    4. Same-candle SL+TP: conservative tiebreak using candle.open proximity

    Returns TradeResult or None if no exit found.
    """
    entry = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit
    direction = signal.direction
    risk = signal.risk_per_unit

    for candle in candles:
        ny_time = candle_ny_time(candle)

        # 1. Session close check
        if is_past_session_close(ny_time, session_cfg):
            exit_price = candle.open
            rr = _calc_rr(direction, entry, exit_price, risk)
            return TradeResult(
                instrument=signal.instrument,
                session=signal.session,
                trade_date=ny_time.date(),
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                exit_price=exit_price,
                exit_reason="time",
                rr_achieved=rr,
                entry_time=signal.signal_time,
                exit_time=candle.time,
            )

        # Determine SL/TP hit on this candle
        if direction == Direction.LONG:
            sl_hit = candle.low <= sl
            tp_hit = candle.high >= tp
        else:
            sl_hit = candle.high >= sl
            tp_hit = candle.low <= tp

        # 2 & 3. Handle exits
        if sl_hit and tp_hit:
            dist_to_sl = abs(candle.open - sl)
            dist_to_tp = abs(candle.open - tp)
            # Conservative: assume the worse outcome (SL hit)
            if dist_to_sl <= dist_to_tp:
                exit_price = sl
                exit_reason = "stop"
            else:
                exit_price = tp
                exit_reason = "target"
        elif sl_hit:
            exit_price = sl
            exit_reason = "stop"
        elif tp_hit:
            exit_price = tp
            exit_reason = "target"
        else:
            continue

        rr = _calc_rr(direction, entry, exit_price, risk)
        return TradeResult(
            instrument=signal.instrument,
            session=signal.session,
            trade_date=candle_ny_time(candle).date(),
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            rr_achieved=rr,
            entry_time=signal.signal_time,
            exit_time=candle.time,
        )

    # If we exhaust candles without exit, close at last candle's close
    if candles:
        last = candles[-1]
        exit_price = last.close
        rr = _calc_rr(direction, entry, exit_price, risk)
        return TradeResult(
            instrument=signal.instrument,
            session=signal.session,
            trade_date=candle_ny_time(last).date(),
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            exit_price=exit_price,
            exit_reason="time",
            rr_achieved=rr,
            entry_time=signal.signal_time,
            exit_time=last.time,
        )

    return None


def _calc_rr(direction: Direction, entry: float, exit_price: float, risk: float) -> float:
    """Calculate RR achieved."""
    if risk <= 0:
        return 0.0
    if direction == Direction.LONG:
        return (exit_price - entry) / risk
    else:
        return (entry - exit_price) / risk


# ─────────────────────────────────────────────────────────────
#  Full simulation for one BE level
# ─────────────────────────────────────────────────────────────


def run_simulation(
    grouped_candles: dict[tuple[SessionName, date], list[Candle]],
    session_configs: dict[SessionName, SessionConfig],
    instrument: str,
    rr_target: float,
    vol_threshold: Optional[float],
    adx_threshold: Optional[float],
    adx_series: dict[datetime, float],
    vol_lookback: int = 20,
    max_trades_per_session: int = 2,
    verbose: bool = False,
) -> SimulationMetrics:
    """Run the full backtest simulation for one filter combination.

    For each (session, date) group:
    1. Build opening range from range-period candles
    2. Detect breakouts in entry window
    3. Apply volume/ADX filters — reject low-conviction breakouts
    4. Simulate trades (fixed SL, no BE)
    5. Allow re-entry after stop-out up to max_trades_per_session

    Returns SimulationMetrics with all trade results.
    """
    metrics = SimulationMetrics(vol_threshold=vol_threshold, adx_threshold=adx_threshold)

    sorted_items = sorted(grouped_candles.items(), key=lambda kv: (kv[0][1], kv[0][0].value))

    for (session_name, session_date), candles in sorted_items:
        if session_name not in session_configs:
            continue

        session_cfg = session_configs[session_name]

        # Separate candles into range period and post-range
        range_candles = []
        post_range_candles = []

        for candle in candles:
            ny_time = candle_ny_time(candle)
            if is_in_range_period(ny_time, session_cfg):
                range_candles.append(candle)
            else:
                post_range_candles.append(candle)

        if not range_candles:
            continue

        # 1. Build opening range
        recorder = RangeRecorder(session_name, instrument)
        for candle in range_candles:
            recorder.add_candle(candle)
        opening_range = recorder.finalize()

        if opening_range is None:
            continue

        # 2. Detect breakouts and simulate trades
        detector = BreakoutDetector(opening_range, rr_target=rr_target)
        trades_taken = 0

        # Walk through post-range candles one at a time
        i = 0
        while i < len(post_range_candles) and trades_taken < max_trades_per_session:
            candle = post_range_candles[i]
            ny_time = candle_ny_time(candle)

            # Only detect breakouts during entry window
            if is_in_entry_window(ny_time, session_cfg):
                signal = detector.check_breakout(candle)
            else:
                signal = None

            if signal:
                # Compute filter values
                prior_candles = range_candles + post_range_candles[:i]
                vol_ratio = compute_volume_ratio(candle, prior_candles, lookback=vol_lookback)
                adx_value = adx_series.get(candle.time)

                # Apply volume filter
                if vol_threshold is not None:
                    if vol_ratio is None or vol_ratio < vol_threshold:
                        detector.reset_direction(signal.direction)
                        i += 1
                        continue

                # Apply ADX filter
                if adx_threshold is not None:
                    if adx_value is None or adx_value < adx_threshold:
                        detector.reset_direction(signal.direction)
                        i += 1
                        continue

                # Both filters pass — simulate trade
                remaining_candles = post_range_candles[i + 1:]
                result = simulate_trade(signal, remaining_candles, session_cfg)

                if result:
                    result.vol_threshold = vol_threshold
                    result.adx_threshold = adx_threshold
                    result.breakout_volume_ratio = vol_ratio
                    result.breakout_adx = adx_value
                    metrics.trades.append(result)
                    trades_taken += 1

                    if verbose:
                        _print_trade(result)

                    # If stopped out, allow re-entry
                    if result.exit_reason == "stop":
                        detector.reset_direction(result.direction)

                        # Advance i past the exit candle
                        exit_time = result.exit_time
                        while i < len(post_range_candles) and post_range_candles[i].time <= exit_time:
                            i += 1
                        continue
                    else:
                        # Target hit or time exit -> done for this session day
                        break

            i += 1

    return metrics


def _print_trade(result: TradeResult) -> None:
    """Print a single trade result for verbose mode."""
    vol_str = f"vol={result.breakout_volume_ratio:.2f}" if result.breakout_volume_ratio is not None else "vol=N/A"
    adx_str = f"ADX={result.breakout_adx:.1f}" if result.breakout_adx is not None else "ADX=N/A"
    print(
        f"  {result.trade_date} {result.session.value:6s} {result.instrument:11s} "
        f"{result.direction.value:5s} entry={result.entry_price:>10.2f} "
        f"sl={result.stop_loss:>10.2f} tp={result.take_profit:>10.2f} "
        f"exit={result.exit_price:>10.2f} reason={result.exit_reason:8s} "
        f"RR={result.rr_achieved:+.2f} {vol_str} {adx_str}"
    )


# ─────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────


def _filter_label(vol: Optional[float], adx: Optional[float]) -> str:
    """Format a filter combo as a short label."""
    v = "Off" if vol is None else f"{vol:.2f}"
    a = "Off" if adx is None else f"{adx:.0f}"
    return f"V={v} A={a}"


def print_filter_grid(
    all_metrics: list[SimulationMetrics],
    vol_thresholds: list[Optional[float]],
    adx_thresholds: list[Optional[float]],
) -> None:
    """Print 2D matrix — ADX rows × Vol columns, cells show +X.XR/N."""
    print("\n" + "=" * 100)
    print("  VOLUME × ADX FILTER GRID  (Total R / Trade Count)")
    print("=" * 100)

    # Build lookup: (vol, adx) -> metrics
    lookup: dict[tuple[Optional[float], Optional[float]], SimulationMetrics] = {}
    for m in all_metrics:
        lookup[(m.vol_threshold, m.adx_threshold)] = m

    # Column headers (volume thresholds)
    vol_labels = ["Off" if v is None else f"{v:.2f}" for v in vol_thresholds]
    col_w = 14
    header = f"{'ADX \\ Vol':>10s} │ " + " │ ".join(f"{l:>{col_w}s}" for l in vol_labels)
    print(header)
    print("─" * (12 + (col_w + 3) * len(vol_thresholds)))

    for adx in adx_thresholds:
        adx_label = "Off" if adx is None else f"{adx:.0f}"
        cells = []
        for vol in vol_thresholds:
            m = lookup.get((vol, adx))
            if m and m.total_trades > 0:
                cells.append(f"{m.total_r:+.1f}R/{m.total_trades}")
            else:
                cells.append("—")
        row = f"{adx_label:>10s} │ " + " │ ".join(f"{c:>{col_w}s}" for c in cells)
        print(row)

    print("─" * (12 + (col_w + 3) * len(vol_thresholds)))


def print_comparison_table(
    all_metrics: list[SimulationMetrics],
) -> tuple[Optional[float], Optional[float]]:
    """Print flat ranked list of all combos sorted by Total R desc.

    Returns (best_vol_threshold, best_adx_threshold).
    """
    print("\n" + "=" * 115)
    print("  VOLUME & ADX FILTER OPTIMIZATION RESULTS  (ranked by Total R)")
    print("=" * 115)

    header = (
        f"{'Vol':>6s} {'ADX':>5s} │ {'Trades':>6s} │ {'Wins':>5s} │ {'Win%':>6s} │ "
        f"{'Avg RR':>7s} │ {'Total R':>8s} │ {'PF':>6s} │ {'MaxDD(R)':>8s} │ {'MaxLoss':>7s}"
    )
    print(header)
    print("─" * 115)

    ranked = sorted(all_metrics, key=lambda m: m.total_r, reverse=True)

    best_vol = None
    best_adx = None
    best_total_r = float("-inf")

    for m in ranked:
        vol_label = "Off" if m.vol_threshold is None else f"{m.vol_threshold:.2f}"
        adx_label = "Off" if m.adx_threshold is None else f"{m.adx_threshold:.0f}"
        line = (
            f"{vol_label:>6s} {adx_label:>5s} │ {m.total_trades:>6d} │ {m.wins:>5d} │ "
            f"{m.win_rate:>5.1%} │ {m.avg_rr:>+7.2f} │ {m.total_r:>+8.2f} │ "
            f"{m.profit_factor:>6.2f} │ {m.max_drawdown_r:>8.2f} │ {m.max_consecutive_losses:>7d}"
        )
        print(line)

        if m.total_r > best_total_r:
            best_total_r = m.total_r
            best_vol = m.vol_threshold
            best_adx = m.adx_threshold

    print("─" * 115)

    return best_vol, best_adx


def print_exit_distribution(all_metrics: list[SimulationMetrics], top_n: int = 5) -> None:
    """Print exit reason distribution for top N configs by Total R."""
    print("\n" + "=" * 80)
    print(f"  EXIT REASON DISTRIBUTION  (top {top_n} configs by Total R)")
    print("=" * 80)

    reasons = ["stop", "target", "time"]
    header = f"{'Vol':>6s} {'ADX':>5s} │ " + " │ ".join(f"{r:>8s}" for r in reasons)
    print(header)
    print("─" * 80)

    ranked = sorted(all_metrics, key=lambda m: m.total_r, reverse=True)[:top_n]

    for m in ranked:
        vol_label = "Off" if m.vol_threshold is None else f"{m.vol_threshold:.2f}"
        adx_label = "Off" if m.adx_threshold is None else f"{m.adx_threshold:.0f}"
        counts = m.exit_reason_counts()
        parts = [f"{counts.get(r, 0):>8d}" for r in reasons]
        print(f"{vol_label:>6s} {adx_label:>5s} │ " + " │ ".join(parts))

    print("─" * 80)


def print_detail_breakdown(
    metrics: SimulationMetrics,
    instruments: list[str],
    sessions: list[SessionName],
) -> None:
    """Print detailed breakdown by instrument × session for one filter config."""
    vol_label = "Off" if metrics.vol_threshold is None else f"{metrics.vol_threshold:.2f}"
    adx_label = "Off" if metrics.adx_threshold is None else f"{metrics.adx_threshold:.0f}"
    print(f"\n{'=' * 90}")
    print(f"  DETAIL BREAKDOWN — Best Config: Vol={vol_label}  ADX={adx_label}")
    print(f"{'=' * 90}")

    header = (
        f"{'Instrument':>11s} │ {'Session':>7s} │ {'Trades':>6s} │ {'Wins':>5s} │ "
        f"{'Win%':>6s} │ {'Avg RR':>7s} │ {'Total R':>8s} │ {'PF':>6s}"
    )
    print(header)
    print("─" * 90)

    for inst in instruments:
        for sess in sessions:
            subset = [t for t in metrics.trades if t.instrument == inst and t.session == sess]
            if not subset:
                continue

            n = len(subset)
            wins = sum(1 for t in subset if t.rr_achieved > 0)
            wr = wins / n if n else 0
            avg_rr = sum(t.rr_achieved for t in subset) / n if n else 0
            total_r = sum(t.rr_achieved for t in subset)

            gross_p = sum(t.rr_achieved for t in subset if t.rr_achieved > 0)
            gross_l = abs(sum(t.rr_achieved for t in subset if t.rr_achieved <= 0))
            pf = gross_p / gross_l if gross_l > 0 else float("inf")

            print(
                f"{inst:>11s} │ {sess.value:>7s} │ {n:>6d} │ {wins:>5d} │ "
                f"{wr:>5.1%} │ {avg_rr:>+7.2f} │ {total_r:>+8.2f} │ {pf:>6.2f}"
            )

    # Totals row
    n = metrics.total_trades
    if n > 0:
        print("─" * 90)
        print(
            f"{'TOTAL':>11s} │ {'ALL':>7s} │ {n:>6d} │ {metrics.wins:>5d} │ "
            f"{metrics.win_rate:>5.1%} │ {metrics.avg_rr:>+7.2f} │ {metrics.total_r:>+8.2f} │ "
            f"{metrics.profit_factor:>6.2f}"
        )

    print("─" * 90)


def print_best_config(
    best_vol: Optional[float],
    best_adx: Optional[float],
    all_metrics: list[SimulationMetrics],
) -> None:
    """Print the best configuration callout with delta vs no-filter baseline."""
    best_metrics = next(
        (m for m in all_metrics if m.vol_threshold == best_vol and m.adx_threshold == best_adx),
        None,
    )
    if not best_metrics:
        return

    vol_label = "Off" if best_vol is None else f"{best_vol:.2f}"
    adx_label = "Off" if best_adx is None else f"{best_adx:.0f}"
    baseline = next(
        (m for m in all_metrics if m.vol_threshold is None and m.adx_threshold is None),
        None,
    )

    print(f"\n{'*' * 60}")
    print(f"  BEST CONFIGURATION: Vol={vol_label}  ADX={adx_label}")
    print(f"{'*' * 60}")
    print(f"  Total R:    {best_metrics.total_r:+.2f}")
    print(f"  Trades:     {best_metrics.total_trades}")
    print(f"  Win Rate:   {best_metrics.win_rate:.1%}")
    print(f"  Avg RR:     {best_metrics.avg_rr:+.2f}")
    print(f"  PF:         {best_metrics.profit_factor:.2f}")
    print(f"  Max DD:     {best_metrics.max_drawdown_r:.2f}R")
    print(f"  Max Consec: {best_metrics.max_consecutive_losses} losses")

    if baseline and (best_vol is not None or best_adx is not None):
        delta_r = best_metrics.total_r - baseline.total_r
        delta_wr = best_metrics.win_rate - baseline.win_rate
        delta_trades = best_metrics.total_trades - baseline.total_trades
        print(f"\n  vs No-Filter baseline (Vol=Off ADX=Off):")
        print(f"    Total R delta:  {delta_r:+.2f}")
        print(f"    Win Rate delta: {delta_wr:+.1%}")
        print(f"    Trade count:    {delta_trades:+d}")

    print(f"{'*' * 60}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────


def _parse_threshold_list(raw: str) -> list[Optional[float]]:
    """Parse a comma-separated threshold list like 'None,1.0,1.5' into [None, 1.0, 1.5]."""
    result: list[Optional[float]] = []
    for token in raw.split(","):
        token = token.strip()
        if token.lower() == "none":
            result.append(None)
        else:
            result.append(float(token))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MS-ORB Volume & ADX Filter Optimization Backtest"
    )
    parser.add_argument(
        "--months", type=int, default=12,
        help="Number of months of historical data (default: 12)",
    )
    parser.add_argument(
        "--instruments", type=str, default="XAU_USD,NAS100_USD",
        help="Comma-separated instruments (default: XAU_USD,NAS100_USD)",
    )
    parser.add_argument(
        "--sessions", type=str, default="TOKYO,LONDON,NY",
        help="Comma-separated sessions (default: TOKYO,LONDON,NY)",
    )
    parser.add_argument(
        "--rr-target", type=float, default=5.0,
        help="Risk-reward target (default: 5.0)",
    )
    parser.add_argument(
        "--vol-filter", type=str, default=None,
        help="Comma-separated volume thresholds (default: None,1.0,1.25,1.5,2.0,2.5)",
    )
    parser.add_argument(
        "--adx-filter", type=str, default=None,
        help="Comma-separated ADX thresholds (default: None,15,20,25,30,35)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-fetch data (ignore cache)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print individual trade details",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse instruments and sessions
    instruments = [s.strip() for s in args.instruments.split(",")]
    session_names = []
    for s in args.sessions.split(","):
        s = s.strip().upper()
        try:
            session_names.append(SessionName(s))
        except ValueError:
            print(f"Unknown session: {s}. Valid: TOKYO, LONDON, NY")
            sys.exit(1)

    session_configs = {name: SESSIONS[name] for name in session_names}
    use_cache = not args.no_cache

    # Parse filter thresholds
    vol_thresholds = _parse_threshold_list(args.vol_filter) if args.vol_filter else DEFAULT_VOL_THRESHOLDS
    adx_thresholds = _parse_threshold_list(args.adx_filter) if args.adx_filter else DEFAULT_ADX_THRESHOLDS

    combos = list(itertools.product(vol_thresholds, adx_thresholds))

    # Date range
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.months * 30)

    print(f"\nMS-ORB Backtest: Volume & ADX Filter Optimization")
    print(f"  Period:      {start_dt.date()} to {end_dt.date()} ({args.months} months)")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Sessions:    {', '.join(s.value for s in session_names)}")
    print(f"  RR Target:   {args.rr_target}")
    print(f"  Vol filters: {['Off' if v is None else v for v in vol_thresholds]}")
    print(f"  ADX filters: {['Off' if a is None else a for a in adx_thresholds]}")
    print(f"  Combos:      {len(combos)}")
    print(f"  Cache:       {'enabled' if use_cache else 'disabled'}")
    print()

    # Initialize OANDA client
    client = OANDAClient()
    if not client.test_connection():
        print("ERROR: Cannot connect to OANDA API. Check your credentials.")
        sys.exit(1)

    # Fetch data for each instrument
    instrument_candles: dict[str, list[Candle]] = {}
    for inst in instruments:
        candles = fetch_candles_range(client, inst, start_dt, end_dt, use_cache=use_cache)
        if not candles:
            print(f"WARNING: No candles fetched for {inst}")
            continue
        instrument_candles[inst] = candles
        print(f"  {inst}: {len(candles)} candles loaded")

    if not instrument_candles:
        print("ERROR: No data available. Exiting.")
        sys.exit(1)

    # Pre-compute ADX series per instrument (once, reused across all combos)
    instrument_adx: dict[str, dict[datetime, float]] = {}
    for inst, candles in instrument_candles.items():
        adx_series = compute_adx_series(candles)
        instrument_adx[inst] = adx_series
        print(f"  {inst}: ADX computed for {len(adx_series)} candles")

    # Group candles by session day (per instrument)
    instrument_groups: dict[str, dict[tuple[SessionName, date], list[Candle]]] = {}
    for inst, candles in instrument_candles.items():
        groups = group_candles_by_session_day(candles, list(session_configs.values()))
        instrument_groups[inst] = groups
        print(f"  {inst}: {len(groups)} session-days identified")

    print()

    # Run simulation for each volume × ADX filter combo
    all_metrics: list[SimulationMetrics] = []

    for combo_idx, (vol_thresh, adx_thresh) in enumerate(combos, 1):
        vol_label = "Off" if vol_thresh is None else f"{vol_thresh:.2f}"
        adx_label = "Off" if adx_thresh is None else f"{adx_thresh:.0f}"
        logger.info(f"[{combo_idx}/{len(combos)}] Simulating Vol={vol_label} ADX={adx_label}")

        combined_metrics = SimulationMetrics(vol_threshold=vol_thresh, adx_threshold=adx_thresh)

        for inst in instruments:
            if inst not in instrument_groups:
                continue

            m = run_simulation(
                grouped_candles=instrument_groups[inst],
                session_configs=session_configs,
                instrument=inst,
                rr_target=args.rr_target,
                vol_threshold=vol_thresh,
                adx_threshold=adx_thresh,
                adx_series=instrument_adx.get(inst, {}),
                max_trades_per_session=TradingConfig.MAX_TRADES_PER_SESSION,
                verbose=args.verbose,
            )
            combined_metrics.trades.extend(m.trades)

        # Sort all trades chronologically for proper drawdown calculation
        combined_metrics.trades.sort(key=lambda t: t.entry_time)
        all_metrics.append(combined_metrics)
        logger.info(
            f"  Vol={vol_label} ADX={adx_label}: "
            f"{combined_metrics.total_trades} trades, Total R={combined_metrics.total_r:+.2f}"
        )

    # Print results
    print_filter_grid(all_metrics, vol_thresholds, adx_thresholds)
    best_vol, best_adx = print_comparison_table(all_metrics)
    print_exit_distribution(all_metrics)

    best_metrics = next(
        (m for m in all_metrics if m.vol_threshold == best_vol and m.adx_threshold == best_adx),
        None,
    )
    if best_metrics:
        print_detail_breakdown(best_metrics, instruments, session_names)

    print_best_config(best_vol, best_adx, all_metrics)


if __name__ == "__main__":
    main()
