"""Calculate trading performance statistics from trade DataFrames."""

import pandas as pd


def summary_stats(df: pd.DataFrame) -> dict:
    """Calculate overall summary statistics.

    Args:
        df: DataFrame of closed trades with rr_achieved column.

    Returns:
        Dict with total_trades, wins, losses, win_rate, total_r, avg_r,
        profit_factor, max_drawdown_r, best_trade, worst_trade.
    """
    if df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    rr = df["rr_achieved"].dropna()
    wins = (rr > 0).sum()
    losses = (rr <= 0).sum()
    total = len(rr)

    winning_r = rr[rr > 0].sum()
    losing_r = abs(rr[rr <= 0].sum())

    return {
        "total_trades": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": wins / total if total > 0 else 0.0,
        "total_r": round(rr.sum(), 2),
        "avg_r": round(rr.mean(), 2) if total > 0 else 0.0,
        "profit_factor": round(winning_r / losing_r, 2) if losing_r > 0 else float("inf") if winning_r > 0 else 0.0,
        "max_drawdown_r": round(_max_drawdown(rr), 2),
        "best_trade": round(rr.max(), 2) if total > 0 else 0.0,
        "worst_trade": round(rr.min(), 2) if total > 0 else 0.0,
    }


def _max_drawdown(rr_series: pd.Series) -> float:
    """Calculate maximum drawdown in R-multiples from a series of rr_achieved values.

    Returns a positive number representing the largest peak-to-trough decline.
    """
    if rr_series.empty:
        return 0.0

    cumulative = rr_series.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    return float(drawdown.max())


def by_instrument(df: pd.DataFrame) -> dict:
    """Calculate stats grouped by instrument.

    Returns:
        Dict mapping instrument name -> summary stats dict.
    """
    if df.empty:
        return {}

    result = {}
    for instrument, group in df.groupby("instrument"):
        result[instrument] = summary_stats(group)
    return result


def by_session(df: pd.DataFrame) -> dict:
    """Calculate stats grouped by session.

    Returns:
        Dict mapping session name -> summary stats dict.
    """
    if df.empty:
        return {}

    result = {}
    for session, group in df.groupby("session"):
        result[session] = summary_stats(group)
    return result


def instrument_session_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cross-tab of instrument x session with win rate and total R.

    Returns:
        DataFrame with MultiIndex columns: (instrument, metric) where metric
        is 'trades', 'win_rate', 'total_r'.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (instrument, session), group in df.groupby(["instrument", "session"]):
        rr = group["rr_achieved"].dropna()
        wins = (rr > 0).sum()
        total = len(rr)
        rows.append({
            "instrument": instrument,
            "session": session,
            "trades": total,
            "win_rate": f"{wins / total * 100:.0f}%" if total > 0 else "0%",
            "total_r": round(rr.sum(), 2),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result


def best_worst_trades(df: pd.DataFrame, n: int = 10) -> dict:
    """Get the top N best and worst trades by rr_achieved.

    Returns:
        Dict with 'best' and 'worst' keys, each containing a list of dicts.
    """
    if df.empty:
        return {"best": [], "worst": []}

    rr_valid = df.dropna(subset=["rr_achieved"]).copy()
    if rr_valid.empty:
        return {"best": [], "worst": []}

    cols = ["instrument", "session", "direction", "entry_time",
            "rr_achieved", "exit_reason", "source"]
    available_cols = [c for c in cols if c in rr_valid.columns]

    best = rr_valid.nlargest(n, "rr_achieved")[available_cols]
    worst = rr_valid.nsmallest(n, "rr_achieved")[available_cols]

    return {
        "best": best.to_dict("records"),
        "worst": worst.to_dict("records"),
    }


def compute_all(df: pd.DataFrame) -> dict:
    """Compute all metrics and return as a single dict.

    Returns:
        Dict with keys: summary, by_instrument, by_session, matrix, best_worst.
    """
    return {
        "summary": summary_stats(df),
        "by_instrument": by_instrument(df),
        "by_session": by_session(df),
        "matrix": instrument_session_matrix(df),
        "best_worst": best_worst_trades(df),
    }
