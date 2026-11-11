"""Logging that writes to console, a per-step file, and an optional
consolidated session file.

The session file exists so a whole pipeline run lands in ONE artefact that can
be attached to an email or an issue, rather than a dozen timestamped files that
have to be stitched together by hand.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

BAR = "=" * 74

_SESSION_HANDLER: logging.Handler | None = None
_SESSION_PATH: Path | None = None


def start_session(log_dir: Path, name: str = "session") -> Path:
    """Open a consolidated log that every subsequent logger also writes to."""
    global _SESSION_HANDLER, _SESSION_PATH
    end_session()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{name}_{stamp}.log"
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(message)s"))
    _SESSION_HANDLER, _SESSION_PATH = h, path
    # attach to loggers already created in this process
    for lg in logging.Logger.manager.loggerDict.values():
        if isinstance(lg, logging.Logger) and lg.name.startswith(("wp", "doctor",
                                                                  "all", "report")):
            lg.addHandler(h)
    return path


def end_session() -> Path | None:
    global _SESSION_HANDLER, _SESSION_PATH
    if _SESSION_HANDLER is not None:
        for lg in logging.Logger.manager.loggerDict.values():
            if isinstance(lg, logging.Logger) and _SESSION_HANDLER in lg.handlers:
                lg.removeHandler(_SESSION_HANDLER)
        _SESSION_HANDLER.close()
    path, _SESSION_HANDLER = _SESSION_PATH, None
    return path


def session_path() -> Path | None:
    return _SESSION_PATH


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(log_dir / f"{name}_{stamp}.log",
                                     encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
            logger.addHandler(fh)
        logger.propagate = False
    if _SESSION_HANDLER is not None and _SESSION_HANDLER not in logger.handlers:
        logger.addHandler(_SESSION_HANDLER)
    return logger


def banner(logger: logging.Logger, text: str) -> None:
    logger.info(BAR)
    logger.info(text)
    logger.info(BAR)
