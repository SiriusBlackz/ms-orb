"""MS-ORB Analysis Dashboard — CLI entry point."""

import argparse
import sys
from pathlib import Path

from db_reader import load_trades
from metrics import compute_all
from charts import generate_all
from report import console_report, csv_export, html_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="MS-ORB Trading Performance Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                               # Console summary
  python main.py --period 30d                   # Last 30 days
  python main.py --instrument TSLA              # TSLA only
  python main.py --session NY                   # NY session only
  python main.py --export all --output-dir out  # Full export
        """,
    )
    parser.add_argument(
        "--period", default="all",
        help="Time period filter: 'all', '7d', '30d', '90d', '1y' (default: all)",
    )
    parser.add_argument(
        "--instrument", default="all",
        help="Instrument filter: 'all', 'TSLA', 'XAU_USD', 'NAS100_USD' (default: all)",
    )
    parser.add_argument(
        "--session", default="all",
        help="Session filter: 'all', 'NY', 'TOKYO', 'LONDON' (default: all)",
    )
    parser.add_argument(
        "--export", choices=["csv", "html", "all"], default=None,
        help="Export format: csv, html, or all (default: console only)",
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="Output directory for exports (default: ./output)",
    )
    parser.add_argument(
        "--oanda-db", default=None,
        help="Path to OANDA trades.db (default: ../oanda_bot/trades.db)",
    )
    parser.add_argument(
        "--ibkr-db", default=None,
        help="Path to IBKR ibkr_trades.db (default: ../ibkr_bot/ibkr_trades.db)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load trades
    print(f"Loading trades (period={args.period}, instrument={args.instrument}, session={args.session})...")
    df = load_trades(
        oanda_db=args.oanda_db,
        ibkr_db=args.ibkr_db,
        period=args.period,
        instrument=args.instrument,
        session=args.session,
    )
    print(f"Found {len(df)} closed trades.")

    # Compute metrics
    metrics = compute_all(df)

    # Always show console report
    console_report(metrics)

    # Always generate chart PNGs
    charts_dir = str(Path(args.output_dir) / "charts")
    print(f"Generating charts -> {charts_dir}/")
    chart_paths = generate_all(df, charts_dir)
    for p in chart_paths:
        print(f"  {p}")

    # Optional exports
    if args.export in ("csv", "all"):
        print(f"\nExporting CSVs -> {args.output_dir}/")
        csv_paths = csv_export(df, metrics, args.output_dir)
        for p in csv_paths:
            print(f"  {p}")

    if args.export in ("html", "all"):
        html_path = str(Path(args.output_dir) / "report.html")
        print(f"\nGenerating HTML report -> {html_path}")
        html_report(df, metrics, charts_dir, html_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
