"""Generate console, CSV, and HTML reports from trade metrics."""

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def console_report(metrics: dict) -> None:
    """Print formatted summary tables to stdout."""
    summary = metrics["summary"]

    # Overall summary
    print("\n" + "=" * 60)
    print("  MS-ORB TRADING PERFORMANCE REPORT")
    print("=" * 60)

    summary_table = [
        ["Total Trades", summary["total_trades"]],
        ["Wins", summary["wins"]],
        ["Losses", summary["losses"]],
        ["Win Rate", f"{summary['win_rate'] * 100:.1f}%"],
        ["Total R", f"{summary['total_r']:+.2f}"],
        ["Average R", f"{summary['avg_r']:+.2f}"],
        ["Profit Factor", f"{summary['profit_factor']:.2f}" if summary["profit_factor"] != float("inf") else "Inf"],
        ["Max Drawdown (R)", f"{summary['max_drawdown_r']:.2f}"],
        ["Best Trade", f"{summary['best_trade']:+.2f}R"],
        ["Worst Trade", f"{summary['worst_trade']:+.2f}R"],
    ]
    print("\n" + tabulate(summary_table, headers=["Metric", "Value"],
                          tablefmt="simple_outline"))

    # By instrument
    by_inst = metrics.get("by_instrument", {})
    if by_inst:
        print("\n--- Performance by Instrument ---")
        inst_rows = []
        for name, stats in by_inst.items():
            inst_rows.append([
                name,
                stats["total_trades"],
                f"{stats['win_rate'] * 100:.0f}%",
                f"{stats['total_r']:+.2f}",
                f"{stats['avg_r']:+.2f}",
                f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "Inf",
            ])
        print(tabulate(inst_rows,
                        headers=["Instrument", "Trades", "Win%", "Total R", "Avg R", "PF"],
                        tablefmt="simple_outline"))

    # By session
    by_sess = metrics.get("by_session", {})
    if by_sess:
        print("\n--- Performance by Session ---")
        sess_rows = []
        for name, stats in by_sess.items():
            sess_rows.append([
                name,
                stats["total_trades"],
                f"{stats['win_rate'] * 100:.0f}%",
                f"{stats['total_r']:+.2f}",
                f"{stats['avg_r']:+.2f}",
                f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "Inf",
            ])
        print(tabulate(sess_rows,
                        headers=["Session", "Trades", "Win%", "Total R", "Avg R", "PF"],
                        tablefmt="simple_outline"))

    # Instrument x Session matrix
    matrix = metrics.get("matrix")
    if matrix is not None and not matrix.empty:
        print("\n--- Instrument x Session Matrix ---")
        print(tabulate(matrix, headers="keys", tablefmt="simple_outline",
                        showindex=False))

    # Best/worst trades
    bw = metrics.get("best_worst", {})
    if bw.get("best"):
        print("\n--- Top Trades ---")
        best_rows = []
        for t in bw["best"][:5]:
            entry = t.get("entry_time", "")
            if hasattr(entry, "strftime"):
                entry = entry.strftime("%Y-%m-%d %H:%M")
            best_rows.append([
                t.get("instrument", ""),
                t.get("session", ""),
                t.get("direction", ""),
                entry,
                f"{t.get('rr_achieved', 0):+.2f}R",
            ])
        print(tabulate(best_rows,
                        headers=["Instrument", "Session", "Dir", "Entry Time", "R"],
                        tablefmt="simple_outline"))

    if bw.get("worst"):
        print("\n--- Worst Trades ---")
        worst_rows = []
        for t in bw["worst"][:5]:
            entry = t.get("entry_time", "")
            if hasattr(entry, "strftime"):
                entry = entry.strftime("%Y-%m-%d %H:%M")
            worst_rows.append([
                t.get("instrument", ""),
                t.get("session", ""),
                t.get("direction", ""),
                entry,
                f"{t.get('rr_achieved', 0):+.2f}R",
            ])
        print(tabulate(worst_rows,
                        headers=["Instrument", "Session", "Dir", "Entry Time", "R"],
                        tablefmt="simple_outline"))

    print()


def csv_export(df: pd.DataFrame, metrics: dict, output_dir: str) -> list[Path]:
    """Export trades and summary to CSV files.

    Returns:
        List of paths to created CSV files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # Trades CSV
    trades_path = out / "trades.csv"
    if not df.empty:
        df.to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)
    paths.append(trades_path)

    # Summary CSV
    summary_path = out / "summary.csv"
    summary = metrics["summary"]
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    paths.append(summary_path)

    # By instrument CSV
    by_inst = metrics.get("by_instrument", {})
    if by_inst:
        inst_path = out / "by_instrument.csv"
        rows = []
        for name, stats in by_inst.items():
            row = {"instrument": name}
            row.update(stats)
            rows.append(row)
        pd.DataFrame(rows).to_csv(inst_path, index=False)
        paths.append(inst_path)

    # By session CSV
    by_sess = metrics.get("by_session", {})
    if by_sess:
        sess_path = out / "by_session.csv"
        rows = []
        for name, stats in by_sess.items():
            row = {"session": name}
            row.update(stats)
            rows.append(row)
        pd.DataFrame(rows).to_csv(sess_path, index=False)
        paths.append(sess_path)

    return paths


def html_report(df: pd.DataFrame, metrics: dict, charts_dir: str,
                output_path: str) -> Path:
    """Generate a self-contained HTML report with embedded charts.

    Args:
        df: Trades DataFrame.
        metrics: Dict from metrics.compute_all().
        charts_dir: Directory containing chart PNGs.
        output_path: Path for the output HTML file.

    Returns:
        Path to the generated HTML file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = metrics["summary"]

    # Embed chart images as base64
    chart_images = {}
    charts_path = Path(charts_dir)
    for png in ["equity_curve.png", "equity_curve_session.png",
                 "r_distribution.png", "monthly_r.png"]:
        fpath = charts_path / png
        if fpath.exists():
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                chart_images[png] = f"data:image/png;base64,{b64}"

    # Build summary table rows
    def fmt_pf(v):
        return "Inf" if v == float("inf") else f"{v:.2f}"

    summary_rows = f"""
    <tr><td>Total Trades</td><td>{summary['total_trades']}</td></tr>
    <tr><td>Wins / Losses</td><td>{summary['wins']} / {summary['losses']}</td></tr>
    <tr><td>Win Rate</td><td>{summary['win_rate'] * 100:.1f}%</td></tr>
    <tr><td>Total R</td><td>{summary['total_r']:+.2f}</td></tr>
    <tr><td>Average R</td><td>{summary['avg_r']:+.2f}</td></tr>
    <tr><td>Profit Factor</td><td>{fmt_pf(summary['profit_factor'])}</td></tr>
    <tr><td>Max Drawdown (R)</td><td>{summary['max_drawdown_r']:.2f}</td></tr>
    <tr><td>Best Trade</td><td>{summary['best_trade']:+.2f}R</td></tr>
    <tr><td>Worst Trade</td><td>{summary['worst_trade']:+.2f}R</td></tr>
    """

    # Build instrument table
    inst_rows = ""
    for name, stats in metrics.get("by_instrument", {}).items():
        inst_rows += f"""
        <tr>
            <td>{name}</td><td>{stats['total_trades']}</td>
            <td>{stats['win_rate'] * 100:.0f}%</td>
            <td>{stats['total_r']:+.2f}</td><td>{stats['avg_r']:+.2f}</td>
            <td>{fmt_pf(stats['profit_factor'])}</td>
        </tr>"""

    # Build session table
    sess_rows = ""
    for name, stats in metrics.get("by_session", {}).items():
        sess_rows += f"""
        <tr>
            <td>{name}</td><td>{stats['total_trades']}</td>
            <td>{stats['win_rate'] * 100:.0f}%</td>
            <td>{stats['total_r']:+.2f}</td><td>{stats['avg_r']:+.2f}</td>
            <td>{fmt_pf(stats['profit_factor'])}</td>
        </tr>"""

    # Chart images HTML
    charts_html = ""
    for name, src in chart_images.items():
        title = name.replace(".png", "").replace("_", " ").title()
        charts_html += f"""
        <div class="chart">
            <h3>{title}</h3>
            <img src="{src}" alt="{title}">
        </div>"""

    # Trades table
    trades_html = ""
    if not df.empty:
        display_cols = ["instrument", "session", "direction", "entry_time",
                        "exit_time", "rr_achieved", "exit_reason", "source"]
        available = [c for c in display_cols if c in df.columns]
        trades_subset = df[available].copy()
        for col in ["entry_time", "exit_time"]:
            if col in trades_subset.columns:
                trades_subset[col] = trades_subset[col].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)
                )
        if "rr_achieved" in trades_subset.columns:
            trades_subset["rr_achieved"] = trades_subset["rr_achieved"].apply(
                lambda x: f"{x:+.2f}" if pd.notna(x) else ""
            )
        trades_html = trades_subset.to_html(index=False, classes="trades-table",
                                             border=0, na_rep="")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MS-ORB Performance Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #1e1e2e; color: #ccccdd; margin: 0; padding: 20px;
    }}
    h1 {{ color: #ffffff; border-bottom: 2px solid #4fc3f7; padding-bottom: 10px; }}
    h2 {{ color: #4fc3f7; margin-top: 30px; }}
    h3 {{ color: #81c784; }}
    table {{ border-collapse: collapse; margin: 10px 0; width: 100%; max-width: 800px; }}
    th {{ background: #2a2a3e; color: #4fc3f7; padding: 8px 12px; text-align: left; }}
    td {{ padding: 6px 12px; border-bottom: 1px solid #333355; }}
    tr:hover {{ background: #2a2a3e; }}
    .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; }}
    .chart {{ margin: 20px 0; }}
    .chart img {{ max-width: 100%; border-radius: 8px; }}
    .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .trades-table {{ font-size: 0.85em; }}
    .footer {{ margin-top: 40px; color: #666688; font-size: 0.8em; }}
    @media (max-width: 900px) {{
        .summary-grid, .charts-grid {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>
<h1>MS-ORB Performance Report</h1>
<p>Generated: {now}</p>

<h2>Summary</h2>
<table>{summary_rows}</table>

<div class="summary-grid">
    <div>
        <h2>By Instrument</h2>
        <table>
            <tr><th>Instrument</th><th>Trades</th><th>Win%</th><th>Total R</th><th>Avg R</th><th>PF</th></tr>
            {inst_rows if inst_rows else '<tr><td colspan="6">No data</td></tr>'}
        </table>
    </div>
    <div>
        <h2>By Session</h2>
        <table>
            <tr><th>Session</th><th>Trades</th><th>Win%</th><th>Total R</th><th>Avg R</th><th>PF</th></tr>
            {sess_rows if sess_rows else '<tr><td colspan="6">No data</td></tr>'}
        </table>
    </div>
</div>

<h2>Charts</h2>
<div class="charts-grid">
    {charts_html if charts_html else '<p>No chart data available.</p>'}
</div>

<h2>Trade Log</h2>
{trades_html if trades_html else '<p>No trades recorded.</p>'}

<div class="footer">
    <p>MS-ORB Analysis Dashboard</p>
</div>
</body>
</html>"""

    out.write_text(html)
    return out
