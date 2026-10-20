"""Error visibility helpers.

Every ``except`` in the pipeline now produces an :class:`ErrorDetail` that
carries the *chain* of causes (e.g. yfinance's own exception wrapped in our
``DataUnavailable``), the full traceback, and a small context dict. The point
is that when something fails, the person running the command can read a real
stack instead of ``Failed: SPY, NVDA`` and guess.
"""
from __future__ import annotations

import dataclasses
import traceback
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorDetail:
    """A single caught failure, with enough info to debug it.

    - ``where`` — free-form label for where it happened (e.g. ``'NVDA'``,
      ``'metadata'``).
    - ``type``  — the exception class name that was *caught* (the wrapper).
    - ``message`` — ``str(exc)`` for that exception.
    - ``cause_chain`` — dotted list of the *underlying* cause types, from
      the wrapper's direct cause outward. For the yfinance case this is
      ``['KeyError', 'OSError', 'Exception']`` (however deep the wrap goes).
    - ``traceback`` — the rendered full traceback (includes the inner frames,
      which is where the actual bug is).
    - ``context`` — a flat dict of whatever the caller wants attached
      (URL, args, request id, ...).
    """

    where: str
    type: str
    message: str
    cause_chain: list[str] = field(default_factory=list)
    traceback: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def format(self, *, max_lines: int | None = None) -> str:
        """Human-readable multi-line block for CLI / logs."""
        lines = [
            f"  [{self.where}] {self.type}: {self.message}",
            f"      caused by: {' → '.join(self.cause_chain) or '-'}",
        ]
        tb = self.traceback or ""
        if max_lines is not None:
            tb = "\n".join(tb.splitlines()[-max_lines:])
        if tb:
            lines.append("      " + tb.replace("\n", "\n      "))
        if self.context:
            ctx = "  ".join(f"{k}={v!r}" for k, v in self.context.items())
            lines.append(f"      context: {ctx}")
        return "\n".join(lines)


def capture(exc: Exception, *, where: str, context: dict[str, Any] | None = None,
            max_causes: int = 10) -> ErrorDetail:
    """Wrap ``exc`` in an :class:`ErrorDetail`, walking ``__cause__`` outward.

    The wrapper is the *caught* exception (e.g. ``DataUnavailable``). The
    ``__cause__`` chain is the underlying library errors that actually
    happened (yfinance's ``KeyError`` etc.). We stop at ``max_causes``
    (default 10) so an infinite self-reference can't hang us.
    """
    chain: list[str] = []
    e: BaseException | None = exc.__cause__
    for _ in range(max_causes):
        if e is None:
            break
        chain.append(type(e).__name__)
        e = e.__cause__

    return ErrorDetail(
        where=str(where),
        type=type(exc).__name__,
        message=str(exc),
        cause_chain=chain,
        traceback=traceback.format_exc(),
        context=dict(context or {}),
    )


def log_detail(logger_name: str, detail: ErrorDetail, *, max_lines: int | None = None) -> None:
    """Log one :class:`ErrorDetail` at ERROR with its full traceback.

    Uses stdlib ``logging`` so existing log configuration (the dev server
    already prints to stderr) picks it up without any extra wiring.
    """
    import logging
    log = logging.getLogger(logger_name or "pytrader.errors")
    log.error(detail.format(max_lines=max_lines))
