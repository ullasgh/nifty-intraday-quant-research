"""Core path, session, and memory settings for nifty_quant."""
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
BARS_1M = DATA_ROOT / "bars" / "1"
BARS_D = DATA_ROOT / "bars" / "D"
FUTURES_ROOT = DATA_ROOT / "futures"
EXTERNAL_ROOT = DATA_ROOT / "external"
MANIFEST_PATH = DATA_ROOT / "MANIFEST.json"
CACHE_ROOT = Path(os.environ.get("NQ_CACHE_ROOT", str(REPO_ROOT / "cache")))
RESULTS_ROOT = Path(os.environ.get("NQ_RESULTS_ROOT", str(REPO_ROOT / "results")))

IST = ZoneInfo("Asia/Kolkata")
SESSION_START = time(9, 15)
SESSION_END = time(15, 29)  # last bar label of the regular session
REGULAR_SESSION_BARS = 375
MEMORY_LIMIT_GB = 4.0
PANEL_VERSION = 1
