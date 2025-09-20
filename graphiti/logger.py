import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def setup_logging() -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # File name like: logs/run_2025-09-18_14-32-10.log
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = logs_dir / f"run_{ts}.log"

    logger = logging.getLogger("graphiti")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs if root configured elsewhere

    # File handler (rotates daily, keeps 7 days)
    file_handler = TimedRotatingFileHandler(
        filename=str(logfile),
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Clear existing handlers if re-run in same process
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Writing to %s", logfile)
    return logger
