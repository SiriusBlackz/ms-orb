"""Generate Matplotlib charts for trade analysis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Consistent dark style
STYLE = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#444466",
    "axes.labelcolor": "#ccccdd",
    "text.color": "#ccccdd",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "grid.color": "#333355",
    "grid.alpha": 0.5,
}

COLORS = ["#4fc3f7", "#ff8a65", "#81c784", "#ce93d8", "#fff176", "#a1887f"]


def _apply_style():
    plt.rcParams.update(STYLE)


def _empty_chart(title: str, output_path: Path):
    """Create a chart with 'No trades yet' message."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "No trades yet", transform=ax.transAxes,
            ha="center", va="center", fontsize=20, color="#666688")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def equity_curve(df: pd.DataFrame, output_dir: str) -> Path:
    """Plot cumulative R over time, separate lines per instrument.

    Args:
        df: DataFrame with entry_time and rr_achieved columns.
        output_dir: Directory to save the PNG.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_dir) / "equity_curve.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df.empty or df["rr_achieved"].dropna().empty:
        _empty_chart("Equity Curve (Cumulative R)", output_path)
        return output_path

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    instruments = df["instrument"].unique()
    for i, inst in enumerate(instruments):
        subset = df[df["instrument"] == inst].sort_values("entry_time").copy()
        subset["cum_r"] = subset["rr_achieved"].cumsum()
        color = COLORS[i % len(COLORS)]
        ax.plot(subset["entry_time"], subset["cum_r"],
                label=inst, color=color, linewidth=1.5, marker="o", markersize=3)

    # Also plot combined
    if len(instruments) > 1:
        combined = df.sort_values("entry_time").copy()
        combined["cum_r"] = combined["rr_achieved"].cumsum()
        ax.plot(combined["entry_time"], combined["cum_r"],
                label="Combined", color="#ffffff", linewidth=2, linestyle="--")

    ax.set_title("Equity Curve (Cumulative R)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative R")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="#666688", linewidth=0.8, linestyle="-")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def equity_curve_by_session(df: pd.DataFrame, output_dir: str) -> Path:
    """Plot cumulative R over time, separate lines per session."""
    output_path = Path(output_dir) / "equity_curve_session.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df.empty or df["rr_achieved"].dropna().empty:
        _empty_chart("Equity Curve by Session", output_path)
        return output_path

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sessions = df["session"].unique()
    for i, sess in enumerate(sessions):
        subset = df[df["session"] == sess].sort_values("entry_time").copy()
        subset["cum_r"] = subset["rr_achieved"].cumsum()
        color = COLORS[i % len(COLORS)]
        ax.plot(subset["entry_time"], subset["cum_r"],
                label=sess, color=color, linewidth=1.5, marker="o", markersize=3)

    ax.set_title("Equity Curve by Session", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative R")
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="#666688", linewidth=0.8, linestyle="-")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def r_distribution(df: pd.DataFrame, output_dir: str) -> Path:
    """Plot histogram of rr_achieved, color-coded by session."""
    output_path = Path(output_dir) / "r_distribution.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df.empty or df["rr_achieved"].dropna().empty:
        _empty_chart("R-Multiple Distribution", output_path)
        return output_path

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sessions = df["session"].unique()
    rr_data = []
    labels = []
    colors = []
    for i, sess in enumerate(sessions):
        subset = df[df["session"] == sess]["rr_achieved"].dropna()
        if not subset.empty:
            rr_data.append(subset.values)
            labels.append(sess)
            colors.append(COLORS[i % len(COLORS)])

    if rr_data:
        ax.hist(rr_data, bins=20, label=labels, color=colors[:len(rr_data)],
                stacked=True, edgecolor="#1e1e2e", linewidth=0.5)

    ax.set_title("R-Multiple Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("R-Multiple")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvline(x=0, color="#ff5555", linewidth=1, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def monthly_r(df: pd.DataFrame, output_dir: str) -> Path:
    """Plot bar chart of monthly R totals."""
    output_path = Path(output_dir) / "monthly_r.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if df.empty or df["rr_achieved"].dropna().empty:
        _empty_chart("Monthly R Performance", output_path)
        return output_path

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    monthly = df.set_index("entry_time").resample("ME")["rr_achieved"].sum()

    bar_colors = ["#81c784" if v >= 0 else "#ff8a65" for v in monthly.values]
    ax.bar(monthly.index, monthly.values, width=20, color=bar_colors,
           edgecolor="#1e1e2e", linewidth=0.5)

    ax.set_title("Monthly R Performance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total R")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="#666688", linewidth=0.8, linestyle="-")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all(df: pd.DataFrame, output_dir: str) -> list[Path]:
    """Generate all charts and return list of file paths."""
    return [
        equity_curve(df, output_dir),
        equity_curve_by_session(df, output_dir),
        r_distribution(df, output_dir),
        monthly_r(df, output_dir),
    ]
