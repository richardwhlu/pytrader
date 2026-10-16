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


if __name__ == "__main__":
    main()
