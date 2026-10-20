"""CLI entry points (thin wrappers around `pytrader.server`)."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="pytrader-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None,
                        help="yaml path (optional; default is .pytrader/config.yaml)")
    args = parser.parse_args()

    cfg = None
    if args.config:
        from pathlib import Path
        import yaml
        from pytrader.config import Config
        cfg = Config.from_dict(yaml.safe_load(Path(args.config).read_text()))

    from pytrader.server import run_cli
    run_cli(host=args.host, port=args.port, cfg=cfg)


def run_backtest_cli() -> None:
    parser = argparse.ArgumentParser(prog="pytrader-backtest")
    parser.add_argument("--name", default="cli-run")
    args = parser.parse_args()
    from pytrader.server import run_backtest_cli as _impl
    _impl(name=args.name)


def run_download_cli() -> None:
    """Collect a universe into the local price cache (rate-limited).

    Example:
      pytrader-download --tickers NVDA,MSFT,GOOG,AAPL,AMD,INTC \
          --lookback-years 5 --batch-size 25
    """
    parser = argparse.ArgumentParser(prog="pytrader-download")
    parser.add_argument("--config", default=None,
                        help="yaml path; universe defaults to the config include list")
    parser.add_argument("--tickers", default=None,
                        help="comma-separated tickers (overrides the config include list)")
    parser.add_argument("--benchmark", default="SPY",
                        help="always include this series as the benchmark")
    parser.add_argument("--lookback-years", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-fetch tickers already in the cache")
    args = parser.parse_args()

    from datetime import date, timedelta
    from pathlib import Path
    from pytrader.config import Config
    from pytrader.data import batch_download
    from pytrader.storage import DuckDBStore

    cfg = None
    if args.config:
        import yaml
        cfg = Config.from_dict(yaml.safe_load(Path(args.config).read_text()))
    cfg = cfg or Config.default()

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else list(cfg.include)
    )
    if args.benchmark and args.benchmark.upper() not in [t.upper() for t in tickers]:
        tickers = [args.benchmark, *tickers]

    db = DuckDBStore(Path(".pytrader") / "pytrader.duckdb")
    start = date.today() - timedelta(days=int(args.lookback_years * 365.25))

    from pytrader.data import YFinanceStore
    upstream = YFinanceStore()
    summary = batch_download(
        upstream, db, tickers, start,
        batch_size=args.batch_size, pause_seconds=args.pause_seconds,
        overwrite=args.overwrite,
    )
    print(f"fetched : {summary['n_fetched']:>4}")
    if summary["fetched"]:
        print("          " + ", ".join(summary["fetched"]))
    print(f"cached  : {summary['n_cached']:>4}  already present")
    print(f"failed  : {summary['n_failed']:>4}")
    if summary.get("failed"):
        print("          " + ", ".join(summary["failed"]))
    if summary.get("failures"):
        # Full traceback for each failure — the thing the user was missing
        # before (they got a list of tickers but no way to debug them).
        from pytrader.diagnostics import ErrorDetail as _ED
        print()
        for d in summary["failures"]:
            if isinstance(d, _ED):
                print(d.format(max_lines=80))
            else:
                print(d)


if __name__ == "__main__":
    main()
