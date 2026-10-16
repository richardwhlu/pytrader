"""pytrader: develop, backtest, and track XGBoost trading strategies.

Module layout (small, replaceable pieces):
- config:     user-managed settings (universe, exclusions, costs)
- data:       price fetchers (yfinance today, swappable later)
- storage:    persistence (duckdb today, postgres later)
- universe:   include/exclude + faith-screening rules
- features:   factor/feature computation
- model:      XGBoost training + persistence
- backtest:   signal -> portfolio -> equity curve (repl. with vectorbt later)
- metrics:    performance statistics
- server:     Flask API + web UI
"""

__version__ = "0.1.0"
