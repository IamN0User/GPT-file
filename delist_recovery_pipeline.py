#!/usr/bin/env python3
"""
Delisting Recovery Pipeline - end-to-end workflow

Overview
--------
This pipeline discovers Item 3.01 delisting notices (8‑K) from the SEC, fetches and
prepares price data around the filing date, engineers features and labels (including
"regained compliance" detection), trains models, and scores candidates.

Rationale & Workflow
--------------------
1) Discover (primary: SEC full‑index master.idx)
   - Iterates quarterly master.idx (QTR1..QTR4) for the requested years.
   - Filters to 8‑K and validates Item 3.01 by fetching the filing index page,
     selecting the primary document (HTML preferred, TXT supported), and verifying
     the actual Item 3.01 section with context keywords.
   - Optional: --price-only keeps only price/market‑value related Item 3.01 events.
   - Optional (default ON): text‑regain enrichment scans forward 8‑Ks in t0..t0+365 (by CIK) to find
     "regained compliance" disclosures using regex over Item 3.01 (preferred) and whole doc.
   - Proxy rotation via proxies.txt is supported for Archives fetches.

2) Fetch Prices (Tiingo)
   - Groups events by ticker, unions windows by ticker, and fetches a single master
     series for each ticker (pre/post window around t0). Per-event slices are written
     from the master. Caching:
       • Per‑ticker master CSV reuse when coverage is sufficient
       • Per‑span CSV for slices
       • Low‑level Tiingo GET cache in data/cache/tiingo (not bypassed by --force)
   - Rate pacing: --rate-sleep seconds between API calls (default 72).

3) Featurize
   - Computes pre/post features around t0:
       • Returns/volatility, 30‑day momentum, drawdowns, trough metrics, slopes, etc.
   - Market‑cap at filing (default cap filter 5e8 USD):
       • market_cap = price_t0 × shares_os (from SEC CompanyFacts at/before t0)
       • If shares unknown → keep event and log a warning
       • Persist shares_os and market_cap; apply --cap-max filter (use -1 to disable)
   - Regained compliance label:
       • Text‑regain: boolean from discover enrichment (regex over Item 3.01 preferred,
         with negative screens; falls back to whole‑doc heuristics)
       • Price‑regain (Nasdaq): close ≥ 1.00 for N consecutive trading days (default N=10),
         only for price‑deficiency events; configurable via --price-threshold and --nasdaq-streak
       • Final label recovered_within_post_days = text_regain OR price_regain; also persists
         recovered_method and timing fields

4) Train
   - Builds ML features (base numeric + reason_conf + one‑hot(reason)).
   - Time‑series CV is robust to small/single‑class folds; training proceeds and saves models.
   - Optional Cox PH survival model if lifelines is installed.

5) Score
   - Recomputes features for events and scores with the trained classifier.
   - Handles single‑class models safely when computing probabilities.

6) Backtest
   - Simulates event-level P&L using cached prices over a test window.
   - BUY at t0 + trough offset (predicted via trough regressor if enabled and available, else observed label, else t0).
   - SELL at BUY + peak offset (predicted via peak regressor if available, else observed label, else fixed hold days).

Key CLI Parameters (by stage)
----------------------------
Common
  --log {DEBUG|INFO|...}: Verbosity
  --force: Rebuild or refetch (stage‑specific semantics)

Discover (search_index)
  discover --years INT --throttle FLOAT [--strict] [--force] [--price-only]
           [--detect-regain-text] [--regain-post-days INT] [--no-proxy]
  - --years: How many years back to scan
  - --throttle: Sleep between SEC requests (s)
  - --strict: Stricter Item 3.01 validation
  - --price-only: Keep only price/market‑value Item 3.01
  - --detect-regain-text: On by default; enrich with text‑regain detection
  - --regain-post-days: Forward window (days) for text detection (default 365)
  - --no-proxy: Force direct connections (ignore proxies.txt)

Discover (full‑index, recommended)
  discover_index --years INT [--throttle FLOAT] [--strict] [--force]
                 [--price-only] [--threads INT] [--log-every-event INT]
                 [--detect-regain-text] [--regain-post-days INT]
  - --threads: Parallel doc fetchers (default 1)
  - --log-every-event: Persist progress every N validations

Fetch Prices (Tiingo)
  fetch_prices --pre-days INT --post-days INT [--force]
               [--rate-sleep FLOAT] [--max-tickers INT]
               [--tiingo-max-requests INT] [--log INFO]
  - --pre-days / --post-days: Window around t0 (defaults 180/365)
  - --rate-sleep: Seconds between API calls (default 72)
  - --force: Ignore per‑span CSV cache (master reuse still applies)
  - --tiingo-max-requests: Optional per‑run cap on Tiingo calls (-1 disables)
    Tiingo free plan Max Request per hour is 50 => 3600/50 = 72 seconds between calls!
  - --max-tickers: Limit distinct tickers (testing)

Featurize
  featurize --pre-days INT --post-days INT [--cap-max FLOAT]
            [--price-threshold FLOAT] [--nasdaq-streak INT] [--log INFO]
  - --cap-max: Drop events with market_cap > cap (default 5e8; -1 disables)
  - --price-threshold: Price for regain rule (default 1.0)
  - --nasdaq-streak: Consecutive days at/above threshold (default 10)

Train
  train [--force] [--log INFO]

Score
  score_universe --mode {cached_events|predict_all} [--log INFO]
  - cached_events: Score the discovered events (predict_all not implemented in this revision)

Backtest
  backtest --test-start YYYY-MM-DD --test-end YYYY-MM-DD \
           [--threshold FLOAT] [--units INT] [--hold-days INT] \
           [--proba-col NAME] [--scored-path PATH] [--use-predicted-trough] [--log INFO]
  - --threshold: Probability cutoff to select events (default 0.6)
  - --units: Units per trade for P&L (default 100)
  - --hold-days: Fallback hold days if no peak prediction/label (default 20)
  - --proba-col: Probability column in scored file (default prob_recovery)
  - --scored-path: Optional explicit scored file; auto-detect newest if omitted
  - --use-predicted-trough: Use trough regressor timing if available; else fallback to label/t0

Environment & Caching
---------------------
• Set TIINGO_API_KEY for Tiingo
• Optional SEC_USER_AGENT to override default UA to reduce 403s
• Proxies: proxies.txt with Webshare format (IP:port:username:password)
• Caches:
  - data/indexes/master_YYYY_QTRn.idx (SEC Archives full‑index)
  - data/prices/* (Tiingo master and per‑span CSVs)
  - data/cache/tiingo/get_*.json (low‑level Tiingo GET cache)
  - data/fundamentals/companyfacts_*.json (SEC CompanyFacts)
  - data/logs/regain_scan.log (regain‑detector traces)
  - data/backtest_report.csv (event-level P&L generated by backtest)

CLI Examples
------------
• Discover via full‑index (recommended):
  python stocks\delist_recovery_pipeline.py discover_index --years 5 --threads 1 --log INFO --force

• Fetch prices for 120/365 windows (pacing for free tier):
  python stocks\delist_recovery_pipeline.py fetch_prices --pre-days 120 --post-days 365 --rate-sleep 72 --log INFO

• Featurize with market‑cap cap and price‑regain defaults:
  python stocks\delist_recovery_pipeline.py featurize --pre-days 120 --post-days 365 --cap-max 5e8 --log INFO

• Train + score:
  python stocks\delist_recovery_pipeline.py train --log INFO
  python stocks\delist_recovery_pipeline.py score_universe --mode cached_events --log INFO

• Backtest over a test window (cached prices only):
  python stocks\delist_recovery_pipeline.py backtest \
      --test-start 2025-07-01 --test-end 2025-09-30 \
      --threshold 0.6 --units 100 --hold-days 20 --log INFO
  # Optional: use predicted trough if trough regressor exists
  python stocks\delist_recovery_pipeline.py backtest \
      --test-start 2025-07-01 --test-end 2025-09-30 \
      --threshold 0.6 --use-predicted-trough

• Backtest with explicit snap tolerance (nearest trading day):
  python stocks\delist_recovery_pipeline.py backtest \
      --test-start 2025-07-01 --test-end 2025-09-30 \
      --threshold 0.6 --units 100 --hold-days 20 --tolerance-days 5 --log INFO

Notes
-----
• Text‑regain detection prefers signals in Item 3.01 and applies negative screens
  (e.g., “has not regained”, forward‑looking intent). It also falls back to
  whole‑document heuristics when the section is missing.
• Price‑regain detection is applied only to price‑deficiency events, using the
  Nasdaq 10‑day rule by default (configurable). NYSE text mentions (802.01C) are
  recognized via regex where present.
• --force on fetch bypasses the per‑span CSV reuse but not the low‑level Tiingo
  HTTP cache; clear data/cache/tiingo to force network re‑fetch.
"""
#      companyfacts (market_cap = price_t0 × shares_os; cap filter --cap-max default 5e8).
 
import os
import sys
import re
import time
import json
import logging
import argparse
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
import io

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from joblib import dump, load
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import xlsxwriter  # noqa: F401
    _XLSX_OK = True
except Exception:
    _XLSX_OK = False

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

# -------------------------
# Small helpers
# -------------------------
def _parse_date_or_none(s):
    if not s:
        return None
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None

# -------------------------
# Config & constants
# -------------------------
logger = logging.getLogger("delist_pipeline")
# Resolve directories relative to this file (repo root = stocks/)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"
MODELS_DIR = DATA_DIR / "models"
EVENTS_FILE = DATA_DIR / "events.json"
TICKERS_FILE = DATA_DIR / "tickers_nasdaq.json"
FEATURES_FILE = DATA_DIR / "features.pkl"
CACHE_DIR = DATA_DIR / "cache"
CACHE_INDEX_FILE = DATA_DIR / "cache_index.json"
INDEX_DIR = DATA_DIR / "indexes"

SEC_HEADERS = {
    "User-Agent": "delist-recovery-pipeline/1.0 (contact: bonnardp@gmail.com)",
    "Accept": "application/json",
}
SEARCH_HEADERS = {
    **SEC_HEADERS,
    "Accept": "application/json",
    "Content-Type": "application/json",
}
SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Optional: override SEC User-Agent via env var to reduce 403s and for testing
_SEC_UA_ENV = os.getenv("SEC_USER_AGENT")
if _SEC_UA_ENV:
    SEC_HEADERS["User-Agent"] = _SEC_UA_ENV
    SEARCH_HEADERS["User-Agent"] = _SEC_UA_ENV

TIINGO_API_KEY = "601d04907c3ea9a9e26d32a2655bbb6366d56cbd" #os.getenv("TIINGO_API_KEY")
TIINGO_BASE = "https://api.tiingo.com"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PRICES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# -------------------------
# Logging setup
# -------------------------
def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Reduce noisy libs if DEBUG not requested
    if lvl > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    # Sanitize unicode in console to avoid Windows encoding issues
    class _AsciiSanitizingFilter(logging.Filter):
        _map = str.maketrans({
            "→": "->", "←": "<-", "±": "+/-", "’": "'", "“": '"', "”": '"', "…": "..."
        })
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                msg2 = msg.translate(self._map)
                # drop any remaining non-ascii characters
                msg2 = msg2.encode("ascii", errors="ignore").decode("ascii")
                record.msg = msg2
                record.args = ()
            except Exception:
                pass
            return True

    logging.getLogger().addFilter(_AsciiSanitizingFilter())

# Dedicated file logger for regain-scan traces (no console spam)
REGAIN_LOGGER: Optional[logging.Logger] = None

def get_regain_logger() -> logging.Logger:
    global REGAIN_LOGGER
    if REGAIN_LOGGER is not None:
        return REGAIN_LOGGER
    lg = logging.getLogger("regain_scan")
    lg.setLevel(logging.INFO)
    # Ensure logs directory exists
    logs_dir = DATA_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / "regain_scan.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    # Apply the same ASCII sanitizing filter to file logs (optional)
    class _AsciiSanitizingFilter(logging.Filter):
        _map = str.maketrans({
            "→": "->", "←": "<-", "±": "+/-", "’": "'", "“": '"', "”": '"', "…": "..."
        })
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                msg2 = msg.translate(self._map)
                record.msg = msg2
                record.args = ()
            except Exception:
                pass
            return True
    lg.addFilter(_AsciiSanitizingFilter())
    REGAIN_LOGGER = lg
    return lg

# -------------------------
# Proxy rotation (Webshare)
# -------------------------
PROXIES_FILE = ROOT_DIR / "proxies.txt"
PROXIES: List[Dict[str, str]] = []

def load_proxies():
    PROXIES.clear()
    if PROXIES_FILE.exists():
        with open(PROXIES_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    ip, port, user, pwd = line.split(":", 3)
                    PROXIES.append({
                        "http": f"http://{user}:{pwd}@{ip}:{port}",
                        "https": f"http://{user}:{pwd}@{ip}:{port}",
                        "_display": f"{ip}:{port}",  # for logging without creds
                    })
                except ValueError:
                    logging.warning("Invalid proxy line (expected IP:port:user:pass): %s", line)

_proxy_idx = 0
def next_proxy() -> Optional[Dict[str, str]]:
    global _proxy_idx
    if not PROXIES:
        return None
    proxy = PROXIES[_proxy_idx % len(PROXIES)]
    _proxy_idx += 1
    return proxy
# ---- No-proxy override (set by CLI)
NO_PROXY = False

# -------------------------
# HTTP helpers with tracing & retry
# -------------------------

class _CachedResponse:
    def __init__(self, text: str):
        self.status_code = 200
        self.content = (text or "").encode("utf-8", errors="ignore")

def _cache_key(url: str) -> Tuple[str, str]:
    """Return (digest, filename) for a given Archives URL."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    # try to keep original extension for readability
    ext = ".html"
    for cand in (".htm", ".html", ".txt", ".xml"):
        if url.lower().endswith(cand):
            ext = cand
            break
    return h, f"{h}{ext}"

def _cache_read(url: str) -> Optional[str]:
    try:
        if not url.lower().startswith("https://www.sec.gov/archives/"):
            return None
        h, fname = _cache_key(url)
        fpath = CACHE_DIR / fname
        if fpath.exists():
            logger.info("Cache hit for Archives: %s -> %s", url, fname)
            return fpath.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", url, e)
    return None

def _cache_write(url: str, text: str):
    try:
        if not url.lower().startswith("https://www.sec.gov/archives/"):
            return
        h, fname = _cache_key(url)
        fpath = CACHE_DIR / fname
        # write file
        fpath.write_text(text or "", encoding="utf-8")
        # update simple index mapping url->filename
        idx = {}
        if CACHE_INDEX_FILE.exists():
            try:
                idx = json.loads(CACHE_INDEX_FILE.read_text(encoding="utf-8"))
            except Exception:
                idx = {}
        idx[url] = {"file": fname, "ts": time.time()}
        CACHE_INDEX_FILE.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        logger.info("Cached Archives content: %s -> %s", url, fname)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", url, e)

def http_post_json(url: str, payload: Dict[str, Any], desc: str,
                   timeout: int = 30, retries: int = 3) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]], Optional[str]]:
    """
    POST JSON with rich tracing. Tries direct first (SEC search-index often blocks DC proxies).
    Writes last payload and headers to data/ for manual replay.
    Honors NO_PROXY (global) to skip proxy attempts.
    """
    DATA_DIR.mkdir(exist_ok=True)
    # Persist the last search payload and headers for manual testing
    try:
        (DATA_DIR / "last_search_payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        (DATA_DIR / "last_search_headers.json").write_text(json.dumps(SEARCH_HEADERS, indent=2, sort_keys=True))
    except Exception as e:
        logger.warning("Failed to write last_search_* files: %s", e)

    # --- Direct attempts
    for attempt in range(1, retries + 1):
        logger.info("POST %s [attempt %d/%d] proxy=%s", desc, attempt, retries, "direct")
        logger.info("  URL: %s", url)
        logger.info("  JSON payload:\n%s", json.dumps(payload, indent=2, sort_keys=True))
        logger.info("  Headers:\n%s", json.dumps(SEARCH_HEADERS, indent=2, sort_keys=True))
        # Build and persist a curl command for replay (direct)
        try:
            _body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _curl_parts = [
                "curl", "-v", "--compressed", "-X", "POST", f"'{url}'"
            ]
            for _k, _v in SEARCH_HEADERS.items():
                _curl_parts += ["-H", f"'{_k}: {_v}'"]
            _curl_parts += ["--data-raw", f"'{_body}'"]
            _curl_cmd = " ".join(_curl_parts)
            logger.info("  CURL (direct):\n%s", _curl_cmd)
            (DATA_DIR / "last_curl_post_direct.sh").write_text(_curl_cmd + "\n", encoding="utf-8")
        except Exception as _e:
            logger.warning("  Failed to write last_curl_post_direct.sh: %s", _e)
        # Write PowerShell replay scripts (curl.exe and Invoke-WebRequest)
        try:
            _body_min = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            # curl.exe PS script
            _ps_curl = (
                "curl.exe -v --compressed -X POST \"" + url + "\" `\n" +
                " `".join([f"-H \"{k}: {v}\"" for k, v in SEARCH_HEADERS.items()]) +
                f" `\n --data-raw '{_body_min}'\n"
            )
            (DATA_DIR / "last_curl_post_direct.ps1").write_text(_ps_curl, encoding="utf-8")
            # Invoke-WebRequest PS script
            _hdr_lines = ";\n".join([f"  '{k}' = '{v}'" for k, v in SEARCH_HEADERS.items()])
            _ps_iwr = (
                "$headers = @{" + "\n" + _hdr_lines + "\n}" +
                f"\n$body = @'\n{_body_min}\n'@\n" +
                f"Invoke-WebRequest -Uri '{url}' -Method POST -Headers $headers -Body $body\n"
            )
            (DATA_DIR / "last_iwr_post_direct.ps1").write_text(_ps_iwr, encoding="utf-8")
        except Exception as _e:
            logger.warning("  Failed to write PowerShell replay scripts: %s", _e)
        try:
            r = requests.post(url, headers=SEARCH_HEADERS, json=payload, timeout=timeout)
            logger.info("  â†’ HTTP %s (%d bytes)", r.status_code, len(r.content))
            if r.status_code != 200:
                try:
                    (DATA_DIR / "last_post_response.txt").write_text(
                        f"STATUS: {r.status_code}\nHEADERS:\n{dict(r.headers)}\n\nBODY:\n" + r.text,
                        encoding="utf-8",
                    )
                except Exception as _e:
                    logger.warning("  Failed to write last_post_response.txt: %s", _e)
            if r.status_code == 200:
                try:
                    return r, r.json(), "direct"
                except Exception as e:
                    logger.warning("  JSON decode error: %s", e)
            elif r.status_code in (403, 429, 503):
                logger.info("  Direct POST blocked (%s).", r.status_code)
                break
            else:
                time.sleep(min(2 * attempt, 5))
        except Exception as e:
            logger.warning("  POST error (direct): %s", e)
            break

    # --- Proxy attempts (skip if NO_PROXY or no proxies)
    if NO_PROXY or not PROXIES:
        if NO_PROXY:
            logger.info("NO_PROXY set: skipping proxy attempts for %s", desc)
        return None, None, None

    for attempt in range(1, retries + 1):
        pxy = next_proxy()
        disp = pxy.get("_display") if pxy else "direct"
        logger.info("POST %s [proxy attempt %d/%d] proxy=%s", desc, attempt, retries, disp)
        logger.info("  URL: %s", url)
        logger.info("  JSON payload:\n%s", json.dumps(payload, indent=2, sort_keys=True))
        logger.info("  Headers:\n%s", json.dumps(SEARCH_HEADERS, indent=2, sort_keys=True))
        # Build and persist a curl command for replay (proxy)
        try:
            _body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _curl_parts = [
                "curl", "-v", "--compressed", "-X", "POST", f"'{url}'"
            ]
            for _k, _v in SEARCH_HEADERS.items():
                _curl_parts += ["-H", f"'{_k}: {_v}'"]
            if pxy and pxy.get("http"):
                _curl_parts += ["--proxy", f"'{pxy['http']}'"]
            _curl_parts += ["--data-raw", f"'{_body}'"]
            _curl_cmd = " ".join(_curl_parts)
            if pxy and pxy.get("http"):
                _san = re.sub(r"://([^:@]+):([^@]+)@", "://***:***@", _curl_cmd)
                logger.info("  CURL (proxy %s, sanitized):\n%s", disp, _san)
            else:
                logger.info("  CURL (proxy %s):\n%s", disp, _curl_cmd)
            (DATA_DIR / "last_curl_post_proxy.sh").write_text(_curl_cmd + "\n", encoding="utf-8")
        except Exception as _e:
            logger.warning("  Failed to write last_curl_post_proxy.sh: %s", _e)
        # Write PowerShell replay scripts (proxy)
        try:
            _body_min = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _ps_curl = (
                "curl.exe -v --compressed -X POST \"" + url + "\" `\n" +
                " `".join([f"-H \"{k}: {v}\"" for k, v in SEARCH_HEADERS.items()]) +
                (f" `\n --proxy '{pxy['http']}'" if pxy and pxy.get('http') else "") +
                f" `\n --data-raw '{_body_min}'\n"
            )
            (DATA_DIR / "last_curl_post_proxy.ps1").write_text(_ps_curl, encoding="utf-8")
            _hdr_lines = ";\n".join([f"  '{k}' = '{v}'" for k, v in SEARCH_HEADERS.items()])
            _ps_iwr = (
                "$headers = @{" + "\n" + _hdr_lines + "\n}" +
                f"\n$body = @'\n{_body_min}\n'@\n" +
                f"Invoke-WebRequest -Uri '{url}' -Method POST -Headers $headers -Body $body\n"
            )
            (DATA_DIR / "last_iwr_post_proxy.ps1").write_text(_ps_iwr, encoding="utf-8")
        except Exception as _e:
            logger.warning("  Failed to write PowerShell proxy replay scripts: %s", _e)
        try:
            r = requests.post(url, headers=SEARCH_HEADERS, json=payload, proxies=pxy, timeout=timeout)
            logger.info("  â†’ HTTP %s (%d bytes)", r.status_code, len(r.content))
            if r.status_code != 200:
                try:
                    (DATA_DIR / "last_post_response.txt").write_text(
                        f"STATUS: {r.status_code}\nHEADERS:\n{dict(r.headers)}\n\nBODY:\n" + r.text,
                        encoding="utf-8",
                    )
                except Exception as _e:
                    logger.warning("  Failed to write last_post_response.txt: %s", _e)
            if r.status_code == 200:
                try:
                    return r, r.json(), disp
                except Exception as e:
                    logger.warning("  JSON decode error: %s", e)
            else:
                time.sleep(min(2 * attempt, 5))
        except Exception as e:
            logger.warning("  POST error (via %s): %s", disp, e)
            time.sleep(min(2 * attempt, 5))
    return None, None, None

def http_get(url: str, desc: str, timeout: int = 30, retries: int = 3) -> Tuple[Optional[requests.Response], Optional[str], Optional[str]]:
    """
    GET with tracing. Honors NO_PROXY to force direct.
    """
    # Archives cache check (fast path)
    try:
        cached = _cache_read(url)
        if cached is not None:
            return _CachedResponse(cached), cached, "cache"
    except Exception:
        pass
    # Prefer proxies first unless NO_PROXY is set
    if not NO_PROXY and PROXIES:
        for attempt in range(1, retries + 1):
            pxy = next_proxy()
            disp = pxy.get("_display") if pxy else "direct"
            logger.info("GET %s [proxy attempt %d/%d] proxy=%s â†’ %s", desc, attempt, retries, disp, url)
            try:
                r = requests.get(url, headers=SEC_HEADERS, proxies=pxy, timeout=timeout)
                logger.info("  â† HTTP %s (%d bytes)", r.status_code, len(r.content))
                if r.status_code == 200:
                    # Save to cache if Archives
                    try:
                        _cache_write(url, r.text)
                    except Exception:
                        pass
                    return r, r.text, disp
                else:
                    time.sleep(min(2 * attempt, 5))
            except Exception as e:
                logger.warning("  GET error (via %s): %s", disp, e)
                time.sleep(min(2 * attempt, 5))
    # Direct first
    for attempt in range(1, retries + 1):
        logger.info("GET %s [attempt %d/%d] proxy=%s â†’ %s", desc, attempt, retries, "direct", url)
        try:
            # Build and persist a curl command for replay (direct)
            try:
                _curl_parts = ["curl", "-v", "--compressed", "-X", "GET", f"'{url}'"]
                for _k, _v in SEC_HEADERS.items():
                    _curl_parts += ["-H", f"'{_k}: {_v}'"]
                _curl_cmd = " ".join(_curl_parts)
                logger.info("  CURL (direct):\n%s", _curl_cmd)
                (DATA_DIR / "last_curl_get_direct.sh").write_text(_curl_cmd + "\n", encoding="utf-8")
                # Also write PowerShell equivalents
                _ps_curl = (
                    "curl.exe -v --compressed -X GET \"" + url + "\" `\n" +
                    " `".join([f"-H \"{k}: {v}\"" for k, v in SEC_HEADERS.items()]) + "\n"
                )
                (DATA_DIR / "last_curl_get_direct.ps1").write_text(_ps_curl, encoding="utf-8")
                _hdr_lines = ";\n".join([f"  '{k}' = '{v}'" for k, v in SEC_HEADERS.items()])
                _ps_iwr = (
                    "$headers = @{" + "\n" + _hdr_lines + "\n}" +
                    f"\nInvoke-WebRequest -Uri '{url}' -Method GET -Headers $headers\n"
                )
                (DATA_DIR / "last_iwr_get_direct.ps1").write_text(_ps_iwr, encoding="utf-8")
            except Exception as _e:
                logger.warning("  Failed to write last_curl_get_direct.sh: %s", _e)
            r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
            logger.info("  â† HTTP %s (%d bytes)", r.status_code, len(r.content))
            if r.status_code == 200:
                try:
                    _cache_write(url, r.text)
                except Exception:
                    pass
                return r, r.text, "direct"
            else:
                time.sleep(min(2 * attempt, 5))
        except Exception as e:
            logger.warning("  GET error (direct): %s", e)
            break

    # Proxy attempts (skip if NO_PROXY)
    if NO_PROXY:
        logger.info("NO_PROXY is set: skipping proxy attempts for %s", desc)
        return None, None, None

    for attempt in range(1, retries + 1):
        pxy = next_proxy()
        disp = pxy.get("_display") if pxy else "direct"
        logger.info("GET %s [proxy attempt %d/%d] proxy=%s â†’ %s", desc, attempt, retries, disp, url)
        try:
            # Build and persist a curl command for replay (proxy)
            try:
                _curl_parts = ["curl", "-v", "--compressed", "-X", "GET", f"'{url}'"]
                for _k, _v in SEC_HEADERS.items():
                    _curl_parts += ["-H", f"'{_k}: {_v}'"]
                if pxy and pxy.get("http"):
                    _curl_parts += ["--proxy", f"'{pxy['http']}'"]
                _curl_cmd = " ".join(_curl_parts)
                if pxy and pxy.get("http"):
                    _san = re.sub(r"://([^:@]+):([^@]+)@", "://***:***@", _curl_cmd)
                    logger.info("  CURL (proxy %s, sanitized):\n%s", disp, _san)
                else:
                    logger.info("  CURL (proxy %s):\n%s", disp, _curl_cmd)
                (DATA_DIR / "last_curl_get_proxy.sh").write_text(_curl_cmd + "\n", encoding="utf-8")
                # Also write PowerShell equivalents
                _ps_curl = (
                    "curl.exe -v --compressed -X GET \"" + url + "\" `\n" +
                    " `".join([f"-H \"{k}: {v}\"" for k, v in SEC_HEADERS.items()]) +
                    (f" `\n --proxy '{pxy['http']}'\n" if pxy and pxy.get('http') else "\n")
                )
                (DATA_DIR / "last_curl_get_proxy.ps1").write_text(_ps_curl, encoding="utf-8")
                _hdr_lines = ";\n".join([f"  '{k}' = '{v}'" for k, v in SEC_HEADERS.items()])
                _ps_iwr = (
                    "$headers = @{" + "\n" + _hdr_lines + "\n}" +
                    f"\nInvoke-WebRequest -Uri '{url}' -Method GET -Headers $headers\n"
                )
                (DATA_DIR / "last_iwr_get_proxy.ps1").write_text(_ps_iwr, encoding="utf-8")
            except Exception as _e:
                logger.warning("  Failed to write last_curl_get_proxy.sh: %s", _e)
            r = requests.get(url, headers=SEC_HEADERS, proxies=pxy, timeout=timeout)
            logger.info("  â† HTTP %s (%d bytes)", r.status_code, len(r.content))
            if r.status_code == 200:
                return r, r.text, disp
            else:
                time.sleep(min(2 * attempt, 5))
        except Exception as e:
            logger.warning("  GET error (via %s): %s", disp, e)
            time.sleep(min(2 * attempt, 5))
    return None, None, None


def polite_sleep(seconds=0.1):
    time.sleep(seconds)

def sec_atom_search_item301(since_date: str, until_date: str, page: int = 0, count: int = 100) -> List[Dict[str, str]]:
    """
    Fallback: SEC full-text Atom feed (GET). Returns list of dicts with minimal fields.
    Pagination via 'start' (1-based) and 'count'.
    """
    # SEC's Atom accepts text search and form-type filters
    # Example: https://www.sec.gov/cgi-bin/srch-edgar?text=ITEM%203.01&first=2019&last=2025&form-type=8-K&output=atom&count=100&start=1
    first_year = since_date[:4]
    last_year  = until_date[:4]
    start = page * count + 1
    url = (
        "https://www.sec.gov/cgi-bin/srch-edgar"
        f"?text=ITEM%203.01&first={first_year}&last={last_year}"
        "&form-type=8-K&output=atom"
        f"&count={count}&start={start}"
    )
    r, html, disp = http_get(url, f"ATOM search page {page}")
    if not html:
        return []
    soup = BeautifulSoup(html, "xml")
    entries = []
    for entry in soup.find_all("entry"):
        link = entry.find("link")
        href = link.get("href") if link else None
        title = entry.find("title").get_text(strip=True) if entry.find("title") else ""
        updated = entry.find("updated").get_text(strip=True) if entry.find("updated") else ""
        # Many ATOM links go directly to the index.htm; use as-is
        if href:
            entries.append({"index_url": href, "title": title, "updated": updated})
    logger.info("ATOM page %d â†’ %d entries", page, len(entries))
    return entries

# -------------------------
# SEC helpers
# -------------------------
SEC_API_BASE = "https://data.sec.gov/api/xbrl/companyfacts"

def sec_companyfacts(cik: int) -> dict:
    """Fetch and cache SEC companyfacts for a CIK."""
    cache_dir = DATA_DIR / "fundamentals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"companyfacts_{int(cik):010d}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    url = f"{SEC_API_BASE}/CIK{int(cik):010d}.json"
    hdr = {**SEC_HEADERS, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdr, timeout=30)
    except Exception as e:
        logger.warning("companyfacts error %s: %s", url, e)
        return {}
    if r.status_code != 200:
        logger.warning("companyfacts %s â†’ %s", url, r.status_code)
        return {}
    obj = r.json()
    try:
        cache.write_text(json.dumps(obj))
    except Exception:
        pass
    return obj


def sec_latest_shares_outstanding(cik: int, asof: dt.date) -> Optional[float]:
    """
    Try multiple tags in companyfacts to get shares outstanding at or before 'asof':
    - dei: EntityCommonStockSharesOutstanding
    - us-gaap: CommonStockSharesOutstanding
    Fallback (less ideal): us-gaap WeightedAverageNumberOfSharesOutstandingBasic
    Returns a float (shares) or None.
    """
    facts = sec_companyfacts(int(cik)) or {}
    candidates = [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ]
    best_val = None; best_end = None
    try:
        facts_units = (facts.get("facts") or {})
    except Exception:
        facts_units = {}
    for taxonomy, tag in candidates:
        try:
            series = facts_units[taxonomy][tag]["units"]
        except Exception:
            continue
        for unit_key, arr in series.items():
            for pt in arr:
                try:
                    end = pt.get("end") or pt.get("fy")
                    if not end:
                        continue
                    end_date = dt.date.fromisoformat(end[:10]) if isinstance(end, str) else dt.date(int(end), 12, 31)
                    if end_date <= asof and isinstance(pt.get("val"), (int, float)):
                        if (best_end is None) or (end_date > best_end):
                            best_val = float(pt["val"]); best_end = end_date
                except Exception:
                    continue
    return best_val

# -------------------------
# Regain detection (Text)
# -------------------------
# Core phrases indicating the exchange notified the issuer it has regained compliance
REGAIN_REGEXES = [
    re.compile(r"\b(regained|back in|returned to)\s+compliance\b", re.I),
    re.compile(r"\b(in\s+compliance\s+with)\s+(Nasdaq|NYSE).*?(listing|continued listing)\b", re.I),
    re.compile(r"\bminimum\s+bid\s+price\s+requirement\b.*\b(in|has)\s+compliance\b", re.I),
    re.compile(r"\bNasdaq\b.*\b5550\(a\)\(2\)\b", re.I),
    re.compile(r"\bNYSE\b.*\b802\.01C\b", re.I),
]

# Item 3.01-focused positive patterns (DOTALL for robustness)
RE_REGAINED_DIRECT = re.compile(
    r"\bhas\s+regained\s+compliance\s+with\s+(?:the\s+)?(?:(?:Nasdaq|NYSE(?:\s+American)?)\s+Listing\s+Rule\s+)?(?P<rule>[0-9A-Za-z\.\(\)]+)",
    re.I | re.S,
)
RE_LETTER_CONFIRMS = re.compile(
    r"\b(?:Nasdaq|NYSE(?:\s+American)?)\b.*?\b(?:notified|advised|informed|determined|confirmed|provided\s+written\s+notification)\b.{0,120}?\b(?:has\s+regained\s+compliance|is\s+in\s+compliance)\b",
    re.I | re.S,
)
RE_BACK_IN_COMPLIANCE = re.compile(
    r"\b(?:has\s+)?(?:again\s+|back\s+)?in\s+compliance\b.{0,80}?\b(?:continued\s+listing\s+(?:standards|requirements)|listing\s+standards?)\b",
    re.I | re.S,
)
RE_TEN_DAYS_CONF = re.compile(
    r"\bbid\s+price\b.{0,40}?\$?1(\.00)?\b.{0,80}?\b(?:10|ten)\s+consecutive\s+business\s+days\b",
    re.I | re.S,
)

# Negative screens â€” apply before positives
RE_NEG_HAS_NOT = re.compile(r"\b(has|had|have)\s+not\s+regained\s+compliance\b", re.I)
RE_NEG_NOT_REGAINED = re.compile(r"\bnot\s+regained\s+compliance\b", re.I)
RE_NEG_INTENT = re.compile(r"\b(intend|plan|aim|seek)s?\s+to\s+regain\s+compliance\b", re.I)
RE_NEG_DEADLINE = re.compile(r"\b(until|by)\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}.*?\bto\s+regain\s+compliance\b", re.I)

def _text_regain_match(doc_html: str) -> Optional[Dict[str, Any]]:
    # Try Item 3.01 section specifically
    try:
        section = extract_item_section(doc_html)
    except Exception:
        section = None

    def _snip(src: str, span: tuple) -> str:
        s, e = span
        s = max(0, s - 60); e = min(len(src), e + 60)
        return src[s:e]

    if section and isinstance(section, str) and section.strip():
        # Negative screens first
        for neg in (RE_NEG_HAS_NOT, RE_NEG_NOT_REGAINED, RE_NEG_INTENT, RE_NEG_DEADLINE):
            if neg.search(section):
                return None
        hits = []
        rule = None
        for rx in (RE_REGAINED_DIRECT, RE_LETTER_CONFIRMS, RE_BACK_IN_COMPLIANCE, RE_TEN_DAYS_CONF):
            m = rx.search(section)
            if m:
                hits.append(_snip(section, m.span()))
                gd = m.groupdict() if hasattr(m, 'groupdict') else None
                if gd and gd.get('rule'):
                    rule = gd['rule']
        if hits:
            low = section.lower()
            if rule is None:
                if "5550(a)(2)" in low or "minimum bid" in low:
                    rule = "nasdaq_5550(a)(2)"
                elif "802.01c" in low:
                    rule = "nyse_802.01C"
            return {"phrases": hits, "rule": rule}

    # Fallback to whole-document heuristics
    text = BeautifulSoup(doc_html, "html.parser").get_text(" ", strip=True)
    low = text.lower()
    hits = []
    for rx in REGAIN_REGEXES:
        m = rx.search(low)
        if m:
            hits.append(_snip(text, m.span()))
    if hits:
        rule = None
        if "5550(a)(2)" in low or "minimum bid" in low:
            rule = "nasdaq_5550(a)(2)"
        elif "802.01c" in low:
            rule = "nyse_802.01C"
        return {"phrases": hits, "rule": rule}
    return None

def detect_text_regain_for_event(ev: Dict[str, Any], post_days: int = 365, throttle: float = 0.1) -> Optional[Dict[str, Any]]:
    """Scan quarterly full-index (master.idx) for subsequent 8-K/8-K/A within [t0, t0+post_days]
    and detect "regained compliance" text in the primary filing document. Does not use SEC search-index.
    """
    logf = get_regain_logger()
    cik = ev.get("cik")
    filed_at = ev.get("filingDate")
    if not cik or not filed_at:
        return None
    try:
        t0 = dt.datetime.strptime(filed_at, "%Y-%m-%d").date()
    except Exception:
        return None
    today = dt.date.today()
    end_date = min(t0 + dt.timedelta(days=int(post_days)), today)
    logf.info("event: ticker=%s cik=%s t0=%s window=[%s..%s]", ev.get("ticker"), cik, filed_at, t0.isoformat(), end_date.isoformat())

    def _year_qtr(d: dt.date) -> Tuple[int, int]:
        return d.year, ((d.month - 1) // 3) + 1

    yq_start = _year_qtr(t0)
    yq_end = _year_qtr(end_date)

    def _iter_quarters(yq0: Tuple[int, int], yq1: Tuple[int, int]):
        y, q = yq0
        y_end, q_end = yq1
        while (y < y_end) or (y == y_end and q <= q_end):
            yield y, q
            if q == 4:
                y += 1; q = 1
            else:
                q += 1

    target_cik = int(cik)
    quarters_scanned = []
    for year, qtr in _iter_quarters(yq_start, yq_end):
        logf.info("quarter: %d QTR%d -> loading master.idx", year, qtr)
        content = _download_quarter_master(year, qtr, force=False)
        if not content:
            logf.info("quarter: %d QTR%d -> missing/unavailable master.idx (skip)", year, qtr)
            # Try next quarter; best-effort as some caches may be missing
            continue
        try:
            rows = _parse_master_idx(content)
        except Exception:
            rows = []
        if not rows:
            logf.info("quarter: %d QTR%d -> parsed 0 rows", year, qtr)
            continue
        quarters_scanned.append((year, qtr, len(rows)))
        # Count candidate rows for this CIK in window
        total_cik = 0; total_window = 0
        for rec in rows:
            try:
                form = (rec.get("form") or "").strip().upper()
                if form not in ("8-K", "8-K/A"):
                    continue
                cik_rec = int(rec.get("cik") or 0)
                if cik_rec != target_cik:
                    continue
                total_cik += 1
                filed = rec.get("date") or ""
                if not filed:
                    continue
                try:
                    fd = dt.datetime.strptime(filed, "%Y-%m-%d").date()
                except Exception:
                    continue
                if fd <= t0 or fd > end_date:
                    continue
                total_window += 1
                accession, index_url = _derive_accession_and_urls(rec)
                if not index_url:
                    continue
                logf.info("doc: %d QTR%d cik=%s accession=%s filed=%s", year, qtr, cik_rec, accession, filed)
                r_index, index_html, _ = http_get(index_url, f"index for regain {accession or ''}")
                if r_index is None or r_index.status_code != 200 or not index_html:
                    logf.info("doc: accession=%s -> index fetch failed (status=%s)", accession, getattr(r_index, "status_code", None))
                    polite_sleep(throttle); continue
                # Prefer HTML primary doc, fallback to .txt primary
                doc_candidates = _ranked_html_doc_urls(index_html) or []
                if not doc_candidates:
                    try:
                        soup_idx = BeautifulSoup(index_html, 'html.parser')
                        table = soup_idx.find("table", {"class": "tableFile", "summary": "Document Format Files"})
                        if table:
                            rows_t = table.find_all("tr")
                            for row_t in rows_t[1:]:
                                cols = row_t.find_all("td")
                                if not cols or len(cols) < 4:
                                    continue
                                a = cols[2].find('a')
                                href = a.get('href') if a else None
                                if not href:
                                    continue
                                ttext = (cols[3].get_text(strip=True) or '').upper()
                                low_href = href.lower()
                                if ttext in ("8-K", "8-K/A") and low_href.endswith('.txt'):
                                    doc_url_txt = _normalize_doc_url(href)
                                    if doc_url_txt:
                                        doc_candidates.append(doc_url_txt)
                                        break
                    except Exception:
                        pass
                logf.info("doc: accession=%s -> %d doc candidates", accession, len(doc_candidates))
                found = None
                doc_url = None
                for cand in doc_candidates:
                    r_doc, doc_html, _ = http_get(cand, f"doc for regain {accession or ''}")
                    if r_doc is None or r_doc.status_code != 200 or not doc_html:
                        logf.info("doc: accession=%s -> doc fetch failed (status=%s) url=%s", accession, getattr(r_doc, "status_code", None), cand)
                        continue
                    found = _text_regain_match(doc_html)
                    doc_url = cand
                    if found:
                        logf.info("FOUND: accession=%s filed=%s rule=%s url=%s", accession, filed, found.get("rule"), doc_url)
                        break
                polite_sleep(throttle)
                if found:
                    return {
                        "regained_by_text": True,
                        "regain_accession": accession,
                        "regain_date": filed,
                        "regain_doc_url": doc_url,
                        "regain_rule": found.get("rule"),
                        "regain_phrases": found.get("phrases", [])[:5],
                    }
            except Exception:
                continue
    logf.info("event: ticker=%s cik=%s t0=%s -> no regain found; quarters_scanned=%s", ev.get("ticker"), cik, filed_at, ",".join([f"{y}Q{q}:{n}" for (y,q,n) in quarters_scanned]))
    return None
def download_ticker_map(force=False) -> Dict[str, int]:
    """Download SEC company_tickers.json and map TICKER -> CIK (int)."""
    if TICKERS_FILE.exists() and not force:
        logger.info("Loading cached ticker map from %s", TICKERS_FILE)
        with open(TICKERS_FILE, "r") as f:
            return json.load(f)
    url = "https://www.sec.gov/files/company_tickers.json"
    r, _, _ = http_get(url, "company_tickers.json")
    if r is None or r.status_code != 200:
        raise RuntimeError("Failed to download company_tickers.json from SEC")
    obj = r.json()
    mapping = {}
    for rec in obj.values():
        t = (rec.get("ticker") or "").upper()
        cik = int(rec.get("cik_str") or rec.get("cik") or 0)
        if t and cik:
            mapping[t] = cik
    with open(TICKERS_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved %d tickers to %s", len(mapping), TICKERS_FILE)
    return mapping

def cik_to_ticker_name(cik: int) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort reverse lookup to get (ticker, company name) from CIK."""
    try:
        mapping = download_ticker_map()
        for tk, c in mapping.items():
            if int(c) == int(cik):
                return tk, None
    except Exception:
        pass
    return None, None

# -------------------------
# ITEM 3.01 validation
# -------------------------
ITEM_HEADER_RE = re.compile(r"item\s*3\.01\b", re.IGNORECASE)
CONTEXT_RE = re.compile(r"(notice\s+of\s+delisting|failure\s+to\s+satisfy|non[- ]?compliance|continued\s+listing)", re.IGNORECASE)
NOT_APPL_RE = re.compile(r"not\s+applicable", re.IGNORECASE)

def _normalize_doc_url(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    url = href
    # Make absolute to simplify checks
    if not url.startswith("http"):
        url = ("https://www.sec.gov" + url) if url.startswith("/") else ("https://www.sec.gov" + "/" + url)
    low = url.lower()
    # Normalize inline viewer URLs like https://www.sec.gov/ix?doc=/Archives/...
    if "/ix?" in low:
        i = low.find("doc=")
        if i != -1:
            # Preserve original case from the original url string
            j = url.find("doc=")
            if j != -1:
                doc = url[j+4:]
            else:
                doc = url[i+4:]
            amp = doc.find("&")
            if amp != -1:
                doc = doc[:amp]
            if doc.startswith("/Archives") or doc.startswith("/archives"):
                return "https://www.sec.gov" + doc
    return url

def _ranked_html_doc_urls(index_html: str) -> List[str]:
    # Parse the Document Format Files table and collect ranked HTML links
    soup = BeautifulSoup(index_html, "html.parser")
    table = soup.find("table", {"class": "tableFile", "summary": "Document Format Files"})
    candidates: List[Tuple[int, str]] = []
    if table:
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if not cols or len(cols) < 3:
                continue
            a = cols[2].find("a")
            href = a.get("href") if a else None
            if not href:
                continue
            low_href = href.lower()
            if not (low_href.endswith(".htm") or low_href.endswith(".html")):
                continue
            type_text = cols[3].get_text(strip=True) if len(cols) > 3 else ""
            doc_name = cols[2].get_text(strip=True) or low_href.rsplit("/", 1)[-1]
            type_up = (type_text or "").strip().upper()
            doc_up = (doc_name or "").strip().upper()

            rank = 100
            if type_up in ("8-K", "8-K/A"):
                rank = 1
            elif re.search(r"(?i)8k.*\.html?$", doc_name or ""):
                rank = 2
            elif not doc_up.startswith("EX-"):
                rank = 3
            url = _normalize_doc_url(href)
            if url:
                candidates.append((rank, url))
    candidates.sort(key=lambda x: x[0])
    seen = set(); out: List[str] = []
    for _, u in candidates:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def extract_primary_doc_url(index_html: str, index_url: str) -> Optional[str]:
    def _normalize_doc_url(href: Optional[str]) -> Optional[str]:
        if not href:
            return None
        url = href
        # Make absolute to simplify checks
        if not url.startswith("http"):
            url = ("https://www.sec.gov" + url) if url.startswith("/") else ("https://www.sec.gov" + "/" + url)
        low = url.lower()
        # Normalize inline viewer URLs like https://www.sec.gov/ix?doc=/Archives/...
        if "/ix?" in low:
            i = low.find("doc=")
            if i != -1:
                # Preserve original case from the original url string
                j = url.find("doc=")
                if j != -1:
                    doc = url[j+4:]
                else:
                    # fallback to lower-cased slice if positions differ, unlikely
                    doc = url[i+4:]
                amp = doc.find("&")
                if amp != -1:
                    doc = doc[:amp]
                if doc.startswith("/Archives") or doc.startswith("/archives"):
                    return "https://www.sec.gov" + doc
        return url

    soup = BeautifulSoup(index_html, "html.parser")
    # Try ranked candidates from the Documents table
    ranked = _ranked_html_doc_urls(index_html)
    if ranked:
        return ranked[0]
    # Fallback: first HTML link not containing 'index'
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.lower().endswith((".htm", ".html")) and "index" not in href.lower():
            return _normalize_doc_url(href)
    return None

def validate_item_301(doc_html: str, strict: bool = True) -> bool:
    # Parse HTML to text (strip tags)
    text = BeautifulSoup(doc_html, "html.parser").get_text(" ", strip=True)
    low = text.lower()
    # Must contain ITEM 3.01 header
    if not ITEM_HEADER_RE.search(low):
        return False
    # Must contain one context phrase somewhere in doc (to reduce false positives)
    return CONTEXT_RE.search(low) is not None

# ---------- Item 3.01 section extraction & rule-based reason tagging ----------
# Section boundaries
ITEM_HEADER_ANY = re.compile(r"item\s*3\.0?1\b", re.I)
NEXT_ITEM_HEADER = re.compile(r"item\s*[1-9](?:\.[0-9]{1,2})?\b", re.I)

# Deterministic patterns (high precision)
POS_PRICE_PATTERNS = [
    # Bid price phrasing variants (Nasdaq/NYSE)
    re.compile(r"minimum\s+(?:closing\s+)?bid\s+price", re.I),
    re.compile(r"(?:average\s+)?closing\s+bid\s+price", re.I),
    re.compile(r"\$?1\.00\b", re.I),
    re.compile(r"10\s+consecutive\s+(?:business|trading)\s+days", re.I),
    re.compile(r"30\s+consecutive\s+(?:business|trading)\s+days", re.I),
    re.compile(r"5550\(a\)\(2\)", re.I),   # Nasdaq Capital Market price rule
    re.compile(r"5450\(a\)\(1\)", re.I),   # Nasdaq Global/Global Select price rule
    re.compile(r"802\.01C", re.I),           # NYSE price rule
]
POS_MV_PATTERNS = [
    re.compile(r"market\s+capitalization", re.I),
    re.compile(r"stockholders[â€™']?\s+equity", re.I),  # handle curly apostrophe
    re.compile(r"\$?50,?0?0?0?0?0?0\b", re.I),       # $50,000,000
    re.compile(r"802\.01B", re.I),                    # NYSE market cap/equity
    re.compile(r"5550\(b\)", re.I),                   # Nasdaq equity/MV alternatives
]
NEG_MERGER_PATTERNS = [
    re.compile(r"in\s+connection\s+with\s+the\s+merger", re.I),
    re.compile(r"consummation\s+of\s+the\s+merger", re.I),
    re.compile(r"voluntary\s+delisting", re.I),
    re.compile(r"going\s+private", re.I),
    re.compile(r"file\s+form\s+25", re.I),
]
NEG_OTHER_RULE_PATTERNS = [
    re.compile(r"5250\(c\)\(1\)", re.I),        # Nasdaq filing delinquency
    re.compile(r"periodic\s+filing\s+requirements", re.I),
    re.compile(r"late\s+filing|failure\s+to\s+file", re.I),
]

EXCHANGE_PAT = re.compile(r"nasdaq|nyse(?:\s+american)?", re.I)
RULE_PAT     = re.compile(r"(5550\(a\)\(2\)|5550\(b\)|802\.01[BC])", re.I)
DAYS_30_PAT  = re.compile(r"30\s+consecutive\s+(?:trading|business)\s+days", re.I)
DAYS_180_PAT = re.compile(r"180\s+(?:calendar\s+)?day", re.I)
USD1_PAT     = re.compile(r"\$?1(\.00)?\b", re.I)

@dataclass
class Reason:
    label: str                 # 'price','market_value','delinquency','merger_voluntary','other'
    confidence: float          # 0..1
    fields: Dict[str, Any]     # e.g., {'exchange':'Nasdaq','rule':'5550(a)(2)','deadline_days':180}

def extract_item_section(html: str) -> Optional[str]:
    """Return plain text of the Item 3.01 section, or None if not found."""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.select('br'):
        br.replace_with('\n')
    text = soup.get_text('\n', strip=True)
    low = text.lower()

    m = ITEM_HEADER_ANY.search(low)
    if not m:
        return None
    start = m.start()
    m2 = NEXT_ITEM_HEADER.search(low, pos=start + 8)
    section = text[start: m2.start()] if m2 else text[start:]
    section = re.sub(r"\n{2,}", "\n\n", section)
    return section.strip()

def classify_item301(section_text: str) -> Reason:
    t = section_text
    fields: Dict[str, Any] = {}
    low = t.lower()

    ex = EXCHANGE_PAT.search(low)
    if ex:
        ex_s = ex.group(0).lower()
        fields['exchange'] = 'Nasdaq' if 'nasdaq' in ex_s else ('NYSE American' if 'american' in ex_s else 'NYSE')
    rule = RULE_PAT.search(section_text)
    if rule:
        fields['rule'] = rule.group(1)

    price_hit  = any(p.search(section_text) for p in POS_PRICE_PATTERNS)
    mv_hit     = any(p.search(section_text) for p in POS_MV_PATTERNS)
    merger_hit = any(p.search(section_text) for p in NEG_MERGER_PATTERNS)
    delin_hit  = any(p.search(section_text) for p in NEG_OTHER_RULE_PATTERNS)

    if merger_hit:
        return Reason('merger_voluntary', 0.99, fields)
    if delin_hit and not (price_hit or mv_hit):
        return Reason('delinquency', 0.95, fields)
    if price_hit:
        if USD1_PAT.search(section_text):  fields['threshold'] = '$1.00'
        if DAYS_30_PAT.search(section_text):  fields['lookback_days'] = 30
        if DAYS_180_PAT.search(section_text): fields['deadline_days'] = 180
        return Reason('price', 0.98, fields)
    if mv_hit:
        return Reason('market_value', 0.95, fields)
    return Reason('other', 0.5, fields)
# ---------- end Item 3.01 helpers ----------

# -------------------------
# Submission .txt prefilter (fast drop of non-price events)
# -------------------------
# Strong positive signals (price / market-cap / float / equity deficiencies)
PREF_POS_PRICE_PATTERNS = [
    re.compile(r"nasdaq\s+(listing\s+)?rule\s*5550\(a\)\(2\)", re.I),
    re.compile(r"nyse(\s+american)?\s+section\s*802\.01c", re.I),
    re.compile(r"minimum\s+bid\s+price", re.I),
    re.compile(r"bid\s+price[^\n]{0,100}\$?1(\.00)?", re.I),
    re.compile(r"30\s+consecutive\s+(trading|business)\s+days", re.I),
    re.compile(r"180[- ]calendar[- ]day(?:s)?\s+(?:compliance|period)", re.I),
]
PREF_POS_MV_PATTERNS = [
    re.compile(r"market\s+value\s+of\s+(publicly\s+held\s+shares|listed\s+securities|public\s+float)", re.I),
    re.compile(r"public\s+float", re.I),
    re.compile(r"market\s+capitalization|market\s+cap", re.I),
    re.compile(r"stockholders'?\s+equity", re.I),
    re.compile(r"nasdaq\s+(listing\s+)?rule\s*5550\(b\)", re.I),
    re.compile(r"nyse(\s+american)?\s+section\s*802\.01b", re.I),
]

# Definitive negatives (mergers/voluntary delisting etc.)
PREF_NEG_MERGER_PATTERNS = [
    re.compile(r"in\s+connection\s+with\s+the\s+merger", re.I),
    re.compile(r"\bmerger\b|acquisition|business\s+combination|consummation\s+of\s+the\s+merger", re.I),
    re.compile(r"voluntary\s+delist(ing)?|voluntary\s+deregistration", re.I),
    re.compile(r"form\s+25[^\n]{0,80}in\s+connection\s+with\s+the\s+merger", re.I),
    re.compile(r"tender\s+offer|going\s+private|liquidation|dissolution|redeem(?:ing)?\s+all\s+shares", re.I),
]
# Other non-price rules often not tied to price/market value
PREF_NEG_OTHER_RULE_PATTERNS = [
    re.compile(r"nasdaq\s+rule\s*5250\(c\)\(1\)", re.I),  # late filings
    re.compile(r"nasdaq\s+rule\s*5635", re.I),               # shareholder approval for issuances
    re.compile(r"nasdaq\s+rule\s*5605|5606", re.I),          # governance/diversity
]

def prefilter_submission_text(txt: str) -> Tuple[bool, Dict[str, Any]]:
    """Return (keep, info). Keep if Item 3.01 present AND any strong positive and no definitive negatives."""
    low = txt.lower()
    info: Dict[str, Any] = {"matched": [], "negatives": [], "exchange": None}
    if not ITEM_HEADER_RE.search(low):
        return False, {"reason": "no_item_301"}

    # Exchanges (best-effort)
    if re.search(r"nasdaq", low, re.I):
        info["exchange"] = "nasdaq"
    elif re.search(r"nyse\s+american", low, re.I):
        info["exchange"] = "nyse american"
    elif re.search(r"nyse", low, re.I):
        info["exchange"] = "nyse"

    pos_hits = any(p.search(low) for p in PREF_POS_PRICE_PATTERNS) or any(p.search(low) for p in PREF_POS_MV_PATTERNS)
    for p in PREF_POS_PRICE_PATTERNS + PREF_POS_MV_PATTERNS:
        m = p.search(low)
        if m:
            info["matched"].append(m.group(0)[:80])

    neg_def = False
    for p in PREF_NEG_MERGER_PATTERNS + PREF_NEG_OTHER_RULE_PATTERNS:
        m = p.search(low)
        if m:
            info["negatives"].append(m.group(0)[:80])
            # Mark definitive negatives (merger/voluntary) as blockers
            if p in PREF_NEG_MERGER_PATTERNS:
                neg_def = True

    keep = pos_hits and not neg_def
    if not keep:
        info.setdefault("reason", "no_positive_or_blocked")
    return keep, info

# -------------------------
# Item 3.01 section extraction and classification (post-filter)
# -------------------------
SUPPORT_DEFICIENCY_PATTERNS = [
    re.compile(r"notice\s+of\s+(deficiency|non[- ]?compliance)", re.I),
    re.compile(r"continued\s+listing", re.I),
    re.compile(r"listing\s+qualifications", re.I),
    re.compile(r"regain\s+compliance|cure\s+period", re.I),
    re.compile(r"hearings?\s+panel|panel\s+decision", re.I),
]

def extract_item_301_section(doc_html: str, max_chars: int = 6000) -> str:
    """Return the textual content of Item 3.01 section only (best-effort)."""
    text = BeautifulSoup(doc_html, "html.parser").get_text("\n", strip=True)
    low = text.lower()
    # Find the start of item 3.01
    m = re.search(r"item\s*3\.01\b", low)
    if not m:
        return text[:max_chars]
    start = m.start()
    # Find the next item header after start
    m2 = re.search(r"\n\s*item\s*[0-9]+\.[0-9]+\b", low[m.end():])
    if m2:
        end = m.end() + m2.start()
        section = text[start:end]
    else:
        section = text[start:]
    return section[:max_chars]

def classify_item301_dict(section_text: str) -> Dict[str, Any]:
    """Classify Item 3.01 into categories and capture reasons."""
    low = section_text.lower() if section_text else ""
    info: Dict[str, Any] = {
        "label": None,
        "exchange": None,
        "matched_rules": [],
        "negatives": [],
        "confidence": 0,
        "excerpt": section_text[:400] if section_text else None,
    }
    # Exchange
    if re.search(r"nasdaq", low, re.I):
        info["exchange"] = "nasdaq"
    elif re.search(r"nyse\s+american", low, re.I):
        info["exchange"] = "nyse american"
    elif re.search(r"nyse", low, re.I):
        info["exchange"] = "nyse"

    # Positives
    pos_price_hits = False
    for p in PREF_POS_PRICE_PATTERNS:
        m = p.search(low)
        if m:
            pos_price_hits = True
            info["matched_rules"].append(m.group(0)[:80])
    pos_mv_hits = False
    for p in POS_MV_PATTERNS:
        m = p.search(low)
        if m:
            pos_mv_hits = True
            info["matched_rules"].append(m.group(0)[:80])

    support_hits = sum(1 for p in SUPPORT_DEFICIENCY_PATTERNS if p.search(low))

    # Negatives
    merger_block = False
    for p in NEG_MERGER_PATTERNS:
        m = p.search(low)
        if m:
            merger_block = True
            info["negatives"].append(m.group(0)[:80])
    other_neg = False
    for p in NEG_OTHER_RULE_PATTERNS:
        m = p.search(low)
        if m:
            other_neg = True
            info["negatives"].append(m.group(0)[:80])

    # Labeling
    if merger_block and not (pos_price_hits or pos_mv_hits):
        info["label"] = "merger_voluntary"
        info["confidence"] = -2
    elif pos_price_hits:
        info["label"] = "price_bid_1"
        info["confidence"] = 2 + support_hits
    elif pos_mv_hits:
        info["label"] = "mv_equity_float"
        info["confidence"] = 2 + support_hits
    elif (support_hits > 0) and not other_neg:
        info["label"] = "other_listing_deficiency"
        info["confidence"] = 1
    else:
        info["label"] = "other_non_price"
        info["confidence"] = 0 if not other_neg else -1

    return info

# -------------------------
# Discovery (SEC search API + validation)
# -------------------------
def discover_notices(years: int = 5, size: int = 200, throttle: float = 0.1, force: bool = False, strict: bool = True, max_pages: Optional[int] = None, price_only: bool = False, detect_regain_text: bool = False, regain_post_days: int = 365) -> List[Dict[str, Any]]:
    since_date = (dt.date.today() - dt.timedelta(days=365 * years)).strftime("%Y-%m-%d")
    until_date = dt.date.today().strftime("%Y-%m-%d")

    # Existing events for resume
    existing: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if EVENTS_FILE.exists() and not force:
        try:
            cur = json.loads(EVENTS_FILE.read_text())
            for e in cur:
                existing[(int(e["cik"]), e["accession"])] = e
            logger.info("Loaded %d existing events from %s", len(existing), EVENTS_FILE)
        except Exception:
            pass

    total_validated = 0
    results: Dict[Tuple[int, str], Dict[str, Any]] = dict(existing)

    # Build query; paginate until no more hits
    page_size = max(50, min(500, size))
    payload = {
        "q": "item 3.01",  # keep tight to reduce noise; extend later if needed
        "dateRange": "custom",
        "startdt": since_date,
        "enddt": until_date,
        "forms": ["8-K"],
        "from": 0,
        "size": page_size,
        "sort": [{"filedAt": {"order": "desc"}}],
    }

        # Try JSON search-index first
    page = 0
    used_atom = False

    while True:
        if not used_atom:
            payload["from"] = page * page_size
            desc = f"SEC search (page {page}, from={payload['from']}, size={page_size})"
            r, data, proxy_disp = http_post_json(SEC_SEARCH_URL, payload, desc)
            if not data:
                logger.warning("search-index not available or blocked; switching to ATOM fallback.")
                used_atom = True
                page = 0
                continue

            hits = data.get("hits", {}).get("hits", [])
            total_hits = data.get("hits", {}).get("total", {}).get("value", 0)
            logger.info("Search page %d: %d hits (total ~%d)", page, len(hits), total_hits)
            if not hits:
                break

            # Process hits from JSON API
            for idx, h in enumerate(hits, 1):
                src = h.get("_source", {})
                cik = int(src.get("cik")) if src.get("cik") is not None else None
                accession = src.get("adsh")
                filed_at = src.get("filedAt", "")[:10]
                company = (src.get("displayNames") or [None])[0]
                ticker = (src.get("tickers") or [None])[0]
                index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"

                logger.info("[%d/%d] filedAt=%s  CIK=%s  TICKER=%s  COMPANY=%s", idx, len(hits), filed_at, cik, ticker, company)
                r_index, index_html, _ = http_get(index_url, f"index for {accession}")
                if r_index is None or r_index.status_code != 200:
                    logger.info("  index fetch failed; skipping")
                    continue

                # Try ranked HTML docs with validation fallback
                doc_candidates = _ranked_html_doc_urls(index_html)
                if not doc_candidates:
                    # minimal fallback: any HTML href not containing 'index'
                    soup_tmp = BeautifulSoup(index_html, 'html.parser')
                    for a in soup_tmp.find_all('a'):
                        href = a.get('href', '')
                        if href.lower().endswith(('.htm', '.html')) and 'index' not in href.lower():
                            norm = _normalize_doc_url(href)
                            if norm:
                                doc_candidates.append(norm)
                found_html = None
                doc_url = None
                for cand in doc_candidates:
                    r_doc, doc_html, _ = http_get(cand, f"doc for {accession}")
                    if r_doc is None or r_doc.status_code != 200 or not doc_html:
                        continue
                    if validate_item_301(doc_html, strict=strict):
                        found_html = doc_html; doc_url = cand; break
                logger.info("  primary doc url (validated): %s", doc_url or "NONE")
                if not found_html:
                    polite_sleep(throttle); continue

                # Extract Item 3.01 section & classify reason
                section = extract_item_section(found_html)
                reason_label = reason_conf = None
                reason_fields = {}
                item_excerpt = None
                if section:
                    _r = classify_item301(section)
                    reason_label = _r.label
                    reason_conf  = float(_r.confidence)
                    reason_fields = _r.fields
                    item_excerpt  = section[:2000]
                logger.info("  REASON=%s conf=%.2f fields=%s", reason_label, (reason_conf or 0.0), reason_fields or {})
                if price_only and reason_label not in ("price", "market_value"):
                    polite_sleep(throttle); continue
                # Legacy classification dict retained for backward compatibility
                cls = classify_item301_dict(section) if section else None

                if not ticker:
                    tk, _ = cik_to_ticker_name(cik)
                    ticker = tk

                key = (cik, accession)
                if key not in results:
                    results[key] = {
                        "cik": cik, "ticker": ticker, "company": company, "filingDate": filed_at,
                        "accession": accession, "filing_index_url": index_url, "filing_doc_url": doc_url,
                        "classification": cls,
                        "reason": reason_label,
                        "reason_conf": reason_conf,
                        "reason_fields": reason_fields,
                        "item301_text": item_excerpt,
                    }
                    # Optionally detect text-based regain within post window and enrich event
                    if detect_regain_text:
                        try:
                            enrich = detect_text_regain_for_event(results[key], post_days=regain_post_days, throttle=throttle)
                            if enrich:
                                results[key].update(enrich)
                                logger.info("  REGAIN_TEXT: yes on %s (%s)", enrich.get("regain_date"), enrich.get("regain_rule"))
                        except Exception as _e:
                            logger.info("  REGAIN_TEXT: detection error: %s", _e)
                    total_validated += 1
                    if total_validated % 50 == 0:
                        # Persist global events
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                        # Also persist a per-quarter snapshot for visibility
                        try:
                            # Derive quarter from most recently filed event
                            try:
                                _d_last = dt.datetime.strptime(filed_at, "%Y-%m-%d").date()
                            except Exception:
                                _d_last = dt.date.today()
                            year = _d_last.year
                            qtr = ((_d_last.month - 1) // 3) + 1
                            q_bounds = {1: (1,3), 2: (4,6), 3: (7,9), 4: (10,12)}[qtr]
                            q_events = []
                            for _ev in results.values():
                                fd = _ev.get("filingDate")
                                if not fd:
                                    continue
                                try:
                                    d = dt.datetime.strptime(fd, "%Y-%m-%d").date()
                                except Exception:
                                    continue
                                if d.year == year and q_bounds[0] <= d.month <= q_bounds[1]:
                                    q_events.append(_ev)
                            (DATA_DIR / f"events_{year}_QTR{qtr}_progress.json").write_text(
                                json.dumps(q_events, indent=2, default=str), encoding="utf-8"
                            )
                        except Exception as _e:
                            logger.warning("Failed writing per-quarter progress: %s", _e)
                        logger.info("Persisted %d validated events so far...", total_validated)

                polite_sleep(throttle)

            page += 1
            if payload["from"] + len(hits) >= total_hits:
                break
            polite_sleep(throttle)

        else:
            # ATOM fallback pagination (count=100 per page)
            atom_entries = sec_atom_search_item301(since_date, until_date, page=page, count=100)
            if not atom_entries:
                break
            for i, ent in enumerate(atom_entries, 1):
                index_url = ent["index_url"]
                logger.info("[ATOM %d/%d] index_url=%s  title=%s", i, len(atom_entries), index_url, ent.get("title"))
                r_index, index_html, _ = http_get(index_url, f"index (ATOM)")
                if r_index is None or r_index.status_code != 200:
                    logger.info("  index fetch failed; skipping")
                    continue

                # Try to recover cik/accession from URL
                # .../Archives/edgar/data/{cik}/{adsh_no_dashes}/{adsh}-index.htm
                m = re.search(r"/data/(\\d+)/(\\d+)/([0-9-]+)-index\\.htm", index_url)
                cik = int(m.group(1)) if m else None
                accession = m.group(3) if m else None

                # Try ranked HTML docs with validation fallback (ATOM)
                doc_candidates = _ranked_html_doc_urls(index_html)
                if not doc_candidates:
                    soup_tmp = BeautifulSoup(index_html, 'html.parser')
                    for a in soup_tmp.find_all('a'):
                        href = a.get('href', '')
                        if href.lower().endswith(('.htm', '.html')) and 'index' not in href.lower():
                            norm = _normalize_doc_url(href)
                            if norm:
                                doc_candidates.append(norm)
                found_html = None
                doc_url = None
                for cand in doc_candidates:
                    r_doc, doc_html, _ = http_get(cand, f"doc (ATOM)")
                    if r_doc is None or r_doc.status_code != 200 or not doc_html:
                        continue
                    if validate_item_301(doc_html, strict=strict):
                        found_html = doc_html; doc_url = cand; break
                logger.info("  primary doc url (validated): %s", doc_url or "NONE")
                if not found_html:
                    polite_sleep(throttle); continue

                section = extract_item_section(found_html)
                reason_label = reason_conf = None
                reason_fields = {}
                item_excerpt = None
                if section:
                    _r = classify_item301(section)
                    reason_label = _r.label
                    reason_conf  = float(_r.confidence)
                    reason_fields = _r.fields
                    item_excerpt  = section[:2000]
                logger.info("  REASON=%s conf=%.2f fields=%s", reason_label, (reason_conf or 0.0), reason_fields or {})
                if price_only and reason_label not in ("price", "market_value"):
                    polite_sleep(throttle); continue
                cls = classify_item301_dict(section) if section else None

                filed_at = ent.get("updated", "")[:10]
                ticker = None
                if cik:
                    tk, _ = cik_to_ticker_name(cik); ticker = tk

                key = (cik, accession or index_url)
                if key not in results:
                    results[key] = {
                        "cik": cik, "ticker": ticker, "company": None, "filingDate": filed_at,
                        "accession": accession, "filing_index_url": index_url, "filing_doc_url": doc_url,
                        "classification": cls,
                        "reason": reason_label,
                        "reason_conf": reason_conf,
                        "reason_fields": reason_fields,
                        "item301_text": item_excerpt,
                    }
                    total_validated += 1
                    if total_validated % 25 == 0:
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                        logger.info("Persisted %d validated events so far...", total_validated)

                polite_sleep(throttle)

            page += 1
            if max_pages is not None and page >= max_pages:
                logger.info("Reached max_pages=%d; stopping.", max_pages)
                break
            polite_sleep(throttle)

    # Save final
    final_list = list(results.values())
    json.dump(final_list, open(EVENTS_FILE, "w"), indent=2, default=str)
    logger.info("Discovery finished. Validated %d events. Saved to %s", total_validated, EVENTS_FILE)
    return final_list

# -------------------------
# Discovery via SEC full-index (quarterly master.idx)
# -------------------------
def _download_quarter_master(year: int, qtr: int, force: bool = False) -> Optional[str]:
    """Download and cache full-index master.idx for a given year/quarter. Returns text."""
    assert qtr in (1, 2, 3, 4)
    dest = INDEX_DIR / f"master_{year}_QTR{qtr}.idx"
    if dest.exists() and not force:
        try:
            return dest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    # Try plain .idx first, then .gz if needed
    base = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/master.idx"
    for url in (base, base + ".gz"):
        r, text, _ = http_get(url, f"full-index {year} QTR{qtr}")
        if r is None or r.status_code != 200:
            continue
        if url.endswith(".gz"):
            try:
                import gzip, io
                data = gzip.decompress(r.content).decode("utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Failed to gunzip master.idx: %s", e)
                continue
        else:
            data = text
        try:
            dest.write_text(data, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to cache %s: %s", dest, e)
        return data
    logger.warning("Failed to download master.idx for %d QTR%d", year, qtr)
    return None

def _parse_master_idx(content: str) -> List[Dict[str, str]]:
    """Parse master.idx content. Returns list of dicts with keys: cik, company, form, date, filename."""
    lines = content.splitlines()
    # Skip header until the dashed separator line
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("CIK|") or ln.startswith("----"):
            start = i + 1
            break
    rows = []
    for ln in lines[start:]:
        if not ln.strip() or ln.startswith("--"):
            continue
        if "|" in ln:
            parts = ln.split("|")
            if len(parts) >= 5:
                cik, company, form, date_filed, filename = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip()
                rows.append({
                    "cik": cik, "company": company, "form": form, "date": date_filed, "filename": filename
                })
    return rows

def _derive_accession_and_urls(rec: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """From a master.idx row, derive (accession, filing index URL)."""
    fn = rec.get("filename", "")  # e.g., edgar/data/320193/0000320193-2024-000123.txt
    if not fn or not fn.lower().endswith(".txt"):
        return None, None
    base = fn.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    accession = base
    # Build index URL per SEC convention
    # Directory is everything up to the .txt; replace filename with {adsh}-index.htm
    dir_path = fn.rsplit("/", 1)[0]
    index_url = f"https://www.sec.gov/Archives/{dir_path}/{accession}-index.htm"
    return accession, index_url

def discover_via_index(years: int = 5, throttle: float = 0.1, force: bool = False, strict: bool = True, price_only: bool = False, log_every_event: int = 50, threads: int = 1, detect_regain_text: bool = False, regain_post_days: int = 365) -> List[Dict[str, Any]]:
    """Discover item 3.01 events using the robust quarterly full-index.
    Downloads ~20 index files for 5y instead of daily or efts API.
    """
    today = dt.date.today()
    start_year = today.year - years + 1

    # Load existing results for resume
    existing: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if EVENTS_FILE.exists() and not force:
        try:
            cur = json.loads(EVENTS_FILE.read_text())
            for e in cur:
                if e.get("cik") and e.get("accession"):
                    existing[(int(e["cik"]), e["accession"])] = e
            logger.info("Loaded %d existing events from %s", len(existing), EVENTS_FILE)
        except Exception:
            pass

    results: Dict[Tuple[int, str], Dict[str, Any]] = dict(existing)
    total_validated = 0

    for year in range(start_year, today.year + 1):
        for qtr in (1, 2, 3, 4):
            # Skip future quarters
            if year == today.year and qtr > ((today.month - 1) // 3 + 1):
                continue
            logger.info("Processing full-index %d QTR%d", year, qtr)
            content = _download_quarter_master(year, qtr, force=force)
            if not content:
                continue
            rows = _parse_master_idx(content)
            # Build candidate list: 8-K within window for this quarter (progress uses this size)
            window_start = today - dt.timedelta(days=365 * years)
            candidates: List[Dict[str, str]] = []
            for rr in rows:
                if rr.get("form") != "8-K":
                    continue
                try:
                    fdt = dt.datetime.strptime(rr.get("date", ""), "%Y-%m-%d").date()
                except Exception:
                    continue
                if window_start <= fdt <= today:
                    candidates.append(rr)
            logger.info("%d QTR%d: %d 8-K candidates to scan", year, qtr, len(candidates))
            try:
                cand_out = []
                for rr in candidates:
                    acc, idx_url = _derive_accession_and_urls(rr)
                    cand_out.append({
                        "cik": rr.get("cik"),
                        "company": rr.get("company"),
                        "form": rr.get("form"),
                        "date": rr.get("date"),
                        "filename": rr.get("filename"),
                        "accession": acc,
                        "index_url": idx_url,
                    })
                cand_path = DATA_DIR / f"candidates_{year}_QTR{qtr}.json"
                cand_path.write_text(json.dumps(cand_out, indent=2, default=str), encoding="utf-8")
                logger.info("Saved candidates to %s", str(cand_path.resolve()))
            except Exception as e:
                logger.warning("Failed to write candidates list for %d QTR%d: %s", year, qtr, e)

            processed = 0
            total_cand = len(candidates)
            use_threads = (threads and threads > 1 and not NO_PROXY and bool(PROXIES))
            jobs: List[Dict[str, Any]] = []
            for r in tqdm(candidates, desc=f"{year}Q{qtr} 8-K", unit="filing", leave=False):
                filed_at = r.get("date")
                accession, index_url = _derive_accession_and_urls(r)
                if not accession or not index_url:
                    continue
                try:
                    cik = int(r.get("cik")) if r.get("cik") else None
                except Exception:
                    cik = None

                # Prefilter via index page "Items" sniff: require Item 3.01 in the Items list
                r_index, index_html, _ = http_get(index_url, f"index for {accession}")
                if r_index is None or r_index.status_code != 200:
                    continue
                has_item301 = False
                if index_html:
                    # Robust index-page sniff: search specific summary blocks and then full text
                    try:
                        soup_idx = BeautifulSoup(index_html, "html.parser")
                    except Exception:
                        soup_idx = None

                    NUM_301 = re.compile(r"(?<!\d)3\.0?1(?!\d)")
                    if soup_idx is not None:
                        # Many filing detail pages include an <div class="info"> block with item lines
                        info_blocks = soup_idx.select('div.info')
                        for blk in info_blocks:
                            t = blk.get_text("\n", strip=True).lower()
                            if NUM_301.search(t) or ITEM_HEADER_ANY.search(t):
                                has_item301 = True
                                break
                        # Also look for an Items label cell/value if present
                        if not has_item301:
                            # Search common label/value patterns
                            for lbl in soup_idx.find_all(text=re.compile(r"^\s*items?\s*:?$", re.I)):
                                # check siblings' text
                                try:
                                    sib_txt = lbl.parent.find_next_sibling().get_text(" ", strip=True).lower()
                                    if NUM_301.search(sib_txt):
                                        has_item301 = True
                                        break
                                except Exception:
                                    continue
                    if not has_item301:
                        # Fallback: plain text search across the page
                        try:
                            txt = soup_idx.get_text("\n", strip=True) if soup_idx else index_html
                        except Exception:
                            txt = index_html
                        low_txt = txt.lower()
                        has_item301 = bool(NUM_301.search(low_txt) or ITEM_HEADER_ANY.search(low_txt))
                if not has_item301:
                    polite_sleep(throttle)
                    processed += 1
                    if processed % 50 == 0:
                        try:
                            json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                            q_bounds = {1: (1,3), 2: (4,6), 3: (7,9), 4: (10,12)}[qtr]
                            q_events = []
                            for _ev in results.values():
                                fd = _ev.get("filingDate")
                                if not fd:
                                    continue
                                try:
                                    d = dt.datetime.strptime(fd, "%Y-%m-%d").date()
                                except Exception:
                                    continue
                                if d.year == year and qtr and q_bounds[0] <= d.month <= q_bounds[1]:
                                    q_events.append(_ev)
                            (DATA_DIR / f"events_{year}_QTR{qtr}_progress.json").write_text(
                                json.dumps(q_events, indent=2, default=str), encoding="utf-8"
                            )
                            (DATA_DIR / f"progress_{year}_QTR{qtr}.json").write_text(
                                json.dumps({
                                    "year": year,
                                    "quarter": qtr,
                                    "processed": processed,
                                    "total_candidates": total_cand,
                                    "validated_events_so_far": total_validated
                                }, indent=2),
                                encoding="utf-8"
                            )
                        except Exception as _e:
                            logger.warning("Periodic persistence failed: %s", _e)
                    continue
                # Build candidate doc list
                doc_candidates = _ranked_html_doc_urls(index_html)
                if not doc_candidates:
                    soup_tmp = BeautifulSoup(index_html, 'html.parser')
                    for a in soup_tmp.find_all('a'):
                        href = a.get('href', '')
                        if href.lower().endswith(('.htm', '.html')) and 'index' not in href.lower():
                            norm = _normalize_doc_url(href)
                            if norm:
                                doc_candidates.append(norm)
                if not doc_candidates:
                    continue
                if use_threads:
                    jobs.append({
                        "cik": cik,
                        "filed_at": filed_at,
                        "accession": accession,
                        "index_url": index_url,
                        "docs": doc_candidates,
                        "company": r.get("company"),
                    })
                    processed += 1
                    if processed % 50 == 0:
                        try:
                            json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                            q_bounds = {1: (1,3), 2: (4,6), 3: (7,9), 4: (10,12)}[qtr]
                            q_events = []
                            for _ev in results.values():
                                fd = _ev.get("filingDate")
                                if not fd:
                                    continue
                                try:
                                    d = dt.datetime.strptime(fd, "%Y-%m-%d").date()
                                except Exception:
                                    continue
                                if d.year == year and qtr and q_bounds[0] <= d.month <= q_bounds[1]:
                                    q_events.append(_ev)
                            (DATA_DIR / f"events_{year}_QTR{qtr}_progress.json").write_text(
                                json.dumps(q_events, indent=2, default=str), encoding="utf-8"
                            )
                            (DATA_DIR / f"progress_{year}_QTR{qtr}.json").write_text(
                                json.dumps({
                                    "year": year,
                                    "quarter": qtr,
                                    "processed": processed,
                                    "total_candidates": total_cand,
                                    "validated_events_so_far": total_validated
                                }, indent=2),
                                encoding="utf-8"
                            )
                        except Exception as _e:
                            logger.warning("Periodic persistence failed: %s", _e)
                    continue
                found_html = None
                doc_url = None
                for cand in doc_candidates:
                    r_doc, doc_html, _ = http_get(cand, f"doc for {accession}")
                    if r_doc is None or r_doc.status_code != 200 or not doc_html:
                        continue
                    if validate_item_301(doc_html, strict=strict):
                        found_html = doc_html; doc_url = cand; break
                if not found_html:
                    polite_sleep(throttle)
                    continue

                # Extract Item 3.01 section & classify reason
                section = extract_item_section(found_html)
                reason_label = reason_conf = None
                reason_fields = {}
                item_excerpt = None
                if section:
                    _r = classify_item301(section)
                    reason_label = _r.label
                    reason_conf  = float(_r.confidence)
                    reason_fields = _r.fields
                    item_excerpt  = section[:2000]
                logger.info("  REASON=%s conf=%.2f fields=%s", reason_label, (reason_conf or 0.0), reason_fields or {})
                if price_only and reason_label not in ("price", "market_value"):
                    polite_sleep(throttle)
                    continue
                cls = classify_item301_dict(section) if section else None

                tk = None
                if cik:
                    tk, _ = cik_to_ticker_name(cik)

                key = (cik or -1, accession)
                if key not in results:
                    results[key] = {
                        "cik": cik,
                        "ticker": tk,
                        "company": r.get("company"),
                        "filingDate": filed_at,
                        "accession": accession,
                        "filing_index_url": index_url,
                        "filing_doc_url": doc_url,
                        "prefilter": {"method": "index_items_sniff", "has_item301": True},
                        "classification": cls,
                        "reason": reason_label,
                        "reason_conf": reason_conf,
                        "reason_fields": reason_fields,
                        "item301_text": item_excerpt,
                    }
                    # Single-threaded enrichment: detect text-based regain for this event
                    if detect_regain_text:
                        try:
                            enrich = detect_text_regain_for_event(results[key], post_days=regain_post_days, throttle=throttle)
                            if enrich:
                                results[key].update(enrich)
                                logger.info("  REGAIN_TEXT: yes on %s (%s)", enrich.get("regain_date"), enrich.get("regain_rule"))
                        except Exception as _e:
                            logger.info("  REGAIN_TEXT: detection error: %s", _e)
                    total_validated += 1
                    if log_every_event and total_validated % int(log_every_event) == 0:
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                        logger.info("Persisted %d validated events so far...", total_validated)
                polite_sleep(throttle)

                # Periodic persistence every 50 processed candidates (even if no new validations)
                processed += 1
                if processed % 50 == 0:
                    try:
                        # Persist global events
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                        # Persist per-quarter subset for visibility
                        q_bounds = {1: (1,3), 2: (4,6), 3: (7,9), 4: (10,12)}[qtr]
                        q_events = []
                        for _ev in results.values():
                            fd = _ev.get("filingDate")
                            if not fd:
                                continue
                            try:
                                d = dt.datetime.strptime(fd, "%Y-%m-%d").date()
                            except Exception:
                                continue
                            if d.year == year and q_bounds[0] <= d.month <= q_bounds[1]:
                                q_events.append(_ev)
                        (DATA_DIR / f"events_{year}_QTR{qtr}_progress.json").write_text(
                            json.dumps(q_events, indent=2, default=str), encoding="utf-8"
                        )
                        # Meta progress file for quick monitoring
                        (DATA_DIR / f"progress_{year}_QTR{qtr}.json").write_text(
                            json.dumps({
                                "year": year,
                                "quarter": qtr,
                                "processed": processed,
                                "total_candidates": total_cand,
                                "validated_events_so_far": total_validated
                            }, indent=2),
                            encoding="utf-8"
                        )
                        logger.info(
                            "Progress %d/%d candidates in %d QTR%d; validated so far %d",
                            processed, total_cand, year, qtr, total_validated,
                        )
                    except Exception as _e:
                        logger.warning("Periodic persistence failed: %s", _e)

            # Process queued jobs in parallel if enabled
            if use_threads and jobs:
                lock = threading.Lock()

                def worker(job: Dict[str, Any]):
                    nonlocal total_validated
                    cik = job["cik"]
                    accession = job["accession"]
                    filed_at = job["filed_at"]
                    index_url = job["index_url"]
                    docs = job.get("docs") or []
                    company = job.get("company")

                    found_html = None
                    chosen_url = None
                    for cand in docs:
                        r_doc, doc_html, _ = http_get(cand, f"doc for {accession}")
                        if r_doc is None or r_doc.status_code != 200 or not doc_html:
                            continue
                        if validate_item_301(doc_html, strict=strict):
                            found_html = doc_html; chosen_url = cand; break
                    if not found_html:
                        polite_sleep(throttle)
                        return
                    section = extract_item_section(found_html)
                    reason_label = reason_conf = None
                    reason_fields = {}
                    item_excerpt = None
                    if section:
                        _r = classify_item301(section)
                        reason_label = _r.label
                        reason_conf  = float(_r.confidence)
                        reason_fields = _r.fields
                        item_excerpt  = section[:2000]
                    logger.info("  REASON=%s conf=%.2f fields=%s", reason_label, (reason_conf or 0.0), reason_fields or {})
                    if price_only and reason_label not in ("price", "market_value"):
                        polite_sleep(throttle)
                        return
                    cls = classify_item301_dict(section) if section else None
                    tk = None
                    if cik:
                        tk, _ = cik_to_ticker_name(cik)
                    key = (cik or -1, accession)
                    with lock:
                        if key not in results:
                            results[key] = {
                                "cik": cik,
                                "ticker": tk,
                                "company": company,
                                "filingDate": filed_at,
                                "accession": accession,
                                "filing_index_url": index_url,
                                "filing_doc_url": chosen_url,
                                "prefilter": {"method": "index_items_sniff", "has_item301": True},
                                "classification": cls,
                                "reason": reason_label,
                                "reason_conf": reason_conf,
                                "reason_fields": reason_fields,
                                "item301_text": item_excerpt,
                            }
                            # Optionally detect text regain for this event
                            if detect_regain_text:
                                try:
                                    enrich = detect_text_regain_for_event(results[key], post_days=regain_post_days, throttle=throttle)
                                    if enrich:
                                        results[key].update(enrich)
                                        logger.info("  REGAIN_TEXT: yes on %s (%s)", enrich.get("regain_date"), enrich.get("regain_rule"))
                                except Exception as _e:
                                    logger.info("  REGAIN_TEXT: detection error: %s", _e)
                            total_validated += 1
                            if log_every_event and total_validated % int(log_every_event) == 0:
                                try:
                                    json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                                except Exception as _e:
                                    logger.warning("Failed to persist events at checkpoint: %s", _e)
                    polite_sleep(throttle)

                with ThreadPoolExecutor(max_workers=threads) as ex:
                    list(ex.map(worker, jobs))

            final_list = list(results.values())
    json.dump(final_list, open(EVENTS_FILE, "w"), indent=2, default=str)
    logger.info("Index discovery finished. Validated %d events. Saved to %s", total_validated, EVENTS_FILE)
    return final_list

# -------------------------
# Tiingo price fetching & caching
# -------------------------
def tiingo_price_fetch(ticker: str, start_date: str, end_date: str, force: bool = False) -> Optional[pd.DataFrame]:
    if not TIINGO_API_KEY:
        logger.error("TIINGO_API_KEY not set. Cannot fetch prices.")
        return None

    safe_name = f"{ticker}__{start_date}__{end_date}".replace("/", "-")
    out_path = PRICES_DIR / f"{safe_name}.csv"
    if out_path.exists() and not force:
        try:
            df = pd.read_csv(out_path, parse_dates=["date"], index_col="date")
            try:
                globals()["TIINGO_LAST_NETWORK"] = False
            except Exception:
                pass
            return df
        except Exception:
            pass

    # Optional max-requests cap (safety for free tiers)
    try:
        _tiingo_max = int(globals().get("TIINGO_MAX_REQUESTS", -1) or -1)
    except Exception:
        _tiingo_max = -1
    if _tiingo_max > 0 and int(globals().get("TIINGO_REQUESTS_COUNT", 0)) >= _tiingo_max:
        logger.warning("Tiingo cap reached (TIINGO_MAX_REQUESTS=%d); skipping %s %s..%s",
                       _tiingo_max, ticker, start_date, end_date)
        return None

    # Count this outbound Tiingo call (best-effort)
    try:
        globals()["TIINGO_REQUESTS_COUNT"] = int(globals().get("TIINGO_REQUESTS_COUNT", 0)) + 1
    except Exception:
        globals()["TIINGO_REQUESTS_COUNT"] = 1
    url = f"{TIINGO_BASE}/tiingo/daily/{ticker}/prices"
    params = {"startDate": start_date, "endDate": end_date}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
    # Low-level HTTP cache for Tiingo JSON (manual clean only)
    tiingo_cache_dir = CACHE_DIR / "tiingo"
    tiingo_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        _ckey = hashlib.md5(json.dumps({"u": url, "p": params}, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        _ckey = hashlib.md5(f"{url}|{start_date}|{end_date}".encode("utf-8")).hexdigest()
    cache_file = tiingo_cache_dir / f"get_{_ckey}.json"
    data = None
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            logger.info("Tiingo cache hit %s", cache_file.name)
        except Exception:
            data = None
    if data is not None:
        if not data:
            logger.warning("No price data for %s between %s and %s (cache)", ticker, start_date, end_date)
            return None
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.set_index('date')
        keep = [c for c in ['open','high','low','close','adjClose','volume'] if c in df.columns]
        df = df[keep].rename(columns={'adjClose':'adj_close'})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        logger.info("Saved Tiingo prices to %s (from cache)", out_path)
        try:
            globals()["TIINGO_LAST_NETWORK"] = False
        except Exception:
            pass
        return df
    logger.info("GET Tiingo %s %sâ†’%s", ticker, start_date, end_date)
    # Robust network with simple retries/backoff for transient disconnects
    r = None
    last_err = None
    for attempt in range(1, 5):  # up to 4 attempts
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            logger.info("  attempt %d -> %s (%d bytes)", attempt, r.status_code, len(r.content))
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 8))
                continue
            # Other non-OK statuses: give up
            break
        except Exception as e:
            last_err = e
            logger.warning("Tiingo GET error (attempt %d): %s", attempt, e)
            time.sleep(min(2 ** attempt, 8))
    if r is None:
        logger.error("Tiingo fetch failed for %s: %s", ticker, last_err)
        return None
    if r.status_code != 200:
        logger.error("Tiingo fetch failed for %s: %s %s", ticker, r.status_code, r.text[:200])
        return None
    # Persist low-level JSON cache for future reuse
    try:
        cache_file.write_text(r.text or "")
    except Exception:
        pass
    data = r.json()
    if not data:
        logger.warning("No price data for %s between %s and %s", ticker, start_date, end_date)
        return None
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.set_index('date')
    keep = [c for c in ['open','high','low','close','adjClose','volume'] if c in df.columns]
    df = df[keep].rename(columns={'adjClose':'adj_close'})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    logger.info("Saved Tiingo prices to %s", out_path)
    try:
        globals()["TIINGO_LAST_NETWORK"] = True
    except Exception:
        pass
    return df

# Aggregated helpers for per-ticker master series
def _event_window_for(ev: Dict[str, Any], pre_days: int, post_days: int) -> Tuple[dt.date, dt.date]:
    filing_date = dt.datetime.strptime(ev['filingDate'], "%Y-%m-%d").date()
    start = filing_date - dt.timedelta(days=pre_days)
    end = filing_date + dt.timedelta(days=post_days)
    return start, end

def _master_cache_path(ticker: str, start: dt.date, end: dt.date) -> Path:
    name = f"{ticker}__{start.isoformat()}__{end.isoformat()}__MASTER.csv"
    return PRICES_DIR / name

def _read_master_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            return df.sort_index()
        except Exception as e:
            logger.warning("Failed reading existing master %s: %s", path, e)
    return None

def _tiingo_fetch_master(ticker: str, start: dt.date, end: dt.date, force: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch a master time series for [start, end] from Tiingo.
    Reuse an existing MASTER file if it fully covers the span; otherwise fetch once.
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse: any MASTER file for this ticker that fully covers the desired span
    for cand in sorted(PRICES_DIR.glob(f"{ticker}__*__MASTER.csv")):
        df = _read_master_if_exists(cand)
        if df is None or df.empty:
            continue
        have_start, have_end = df.index.min().date(), df.index.max().date()
        if have_start <= start and have_end >= end:
            logger.info("Reusing existing master for %s covering %sâ†’%s: %s",
                        ticker, have_start, have_end, cand.name)
            return df

    # Else new fetch for required span
    try:
        # Default to False; tiingo_price_fetch will set True when a real network GET occurs,
        # and set False when served from JSON cache.
        globals()["TIINGO_LAST_NETWORK"] = False
    except Exception:
        pass
    df = tiingo_price_fetch(
        ticker,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        force=force
    )
    if df is None or df.empty:
        return None

    out_path = _master_cache_path(ticker, start, end)
    df.sort_index().to_csv(out_path)
    logger.info("Saved master for %s to %s", ticker, out_path.name)
    return df.sort_index()

def _write_event_slice_from_master(ticker: str, ev: Dict[str, Any],
                                   master: pd.DataFrame,
                                   start: dt.date, end: dt.date) -> Path:
    slice_path = PRICES_DIR / f"{ticker}__{start.isoformat()}__{end.isoformat()}.csv"
    keep = [c for c in ['open','high','low','close','adj_close','adjClose','volume'] if c in master.columns]
    df = master.loc[(master.index.date >= start) & (master.index.date <= end)][keep].copy()
    if 'adjClose' in df.columns and 'adj_close' not in df.columns:
        df = df.rename(columns={'adjClose':'adj_close'})
    df.to_csv(slice_path)
    return slice_path

def fetch_prices_for_events(events: List[Dict[str, Any]],
                            pre_days: int = 180, post_days: int = 365,
                            force: bool = False,
                            rate_sleep: float = 0.0):
    """
    Aggregated Tiingo fetching:
      - Group events by ticker.
      - Compute union window [min_start, max_end] per ticker.
      - Fetch ONE master series per ticker and write per-event slices from it.
    Downstream (featurize/train) continues to use per-event 'price_slice' CSVs.
    """
    # Build per-ticker groups and spans
    ticker_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    span_by_ticker: Dict[str, Tuple[dt.date, dt.date]] = {}

    for ev in events:
        ticker = ev.get("ticker")
        if not ticker or not ev.get("filingDate"):
            continue
        start, end = _event_window_for(ev, pre_days, post_days)
        ticker_events[ticker].append(ev)
        if ticker not in span_by_ticker:
            span_by_ticker[ticker] = (start, end)
        else:
            cur_start, cur_end = span_by_ticker[ticker]
            span_by_ticker[ticker] = (min(cur_start, start), max(cur_end, end))

    updated_events = []
    total_tickers = len(ticker_events)
    done_tickers = 0
    for i, (ticker, evs) in enumerate(ticker_events.items(), 1):
        min_start, max_end = span_by_ticker[ticker]
        logger.info("Ticker %s: consolidated span %s â†’ %s over %d events",
                    ticker, min_start.isoformat(), max_end.isoformat(), len(evs))

        master = _tiingo_fetch_master(ticker, min_start, max_end, force=force)
        if master is None or master.empty:
            logger.warning("No data for %s; skipping its %d events", ticker, len(evs))
            continue
        # Progress: mark a ticker done once its master series is available (fetched or reused)
        done_tickers += 1
        try:
            pct = (100.0 * done_tickers / total_tickers) if total_tickers else 100.0
        except Exception:
            pct = 0.0
        logger.info("fetch progress: %d/%d (%.1f%%) tickers processed", done_tickers, total_tickers, pct)

        for ev in evs:
            start, end = _event_window_for(ev, pre_days, post_days)
            slice_path = _write_event_slice_from_master(ticker, ev, master, start, end)
            ev_mod = dict(ev)
            ev_mod['price_slice'] = str(slice_path)
            updated_events.append(ev_mod)

        if rate_sleep and rate_sleep > 0:
            try:
                _did_network = bool(globals().get("TIINGO_LAST_NETWORK", False))
            except Exception:
                _did_network = True
            if _did_network:
                time.sleep(rate_sleep)
            try:
                globals()["TIINGO_LAST_NETWORK"] = False
            except Exception:
                pass

    json.dump(updated_events, open(EVENTS_FILE, "w"), indent=2, default=str)
    logger.info("fetch_prices: updated %d events across %d tickers", len(updated_events), len(ticker_events))
    return updated_events

# -------------------------
# Feature engineering
# -------------------------
def compute_event_features(event: Dict[str, Any], pre_window_days: int = 180, post_window_days: int = 365) -> Optional[Dict[str, Any]]:
    price_path = event.get('price_slice')
    if not price_path or not Path(price_path).exists():
        return None
    try:
        df = pd.read_csv(price_path, parse_dates=["date"], index_col="date")
    except Exception:
        try:
            df = pd.read_csv(price_path, parse_dates=[0], index_col=0)
        except Exception as e:
            logger.error("Failed reading price slice %s: %s", price_path, e)
            return None

    price_col = 'adj_close' if 'adj_close' in df.columns else ('close' if 'close' in df.columns else None)
    if not price_col:
        return None

    df = df.sort_index()
    t0 = pd.Timestamp(dt.datetime.strptime(event['filingDate'], "%Y-%m-%d"))
    pre = df.loc[(df.index >= t0 - pd.Timedelta(days=pre_window_days)) & (df.index < t0)].copy()
    post = df.loc[(df.index >= t0) & (df.index <= t0 + pd.Timedelta(days=post_window_days))].copy()
    # No-leakage guards (pre strictly < t0; post >= t0)
    assert True if pre.empty else (pre.index.max() < t0), "pre window must be strictly before t0"
    assert True if post.empty else (post.index.min() >= t0), "post window must start at/after t0"
    if pre.empty or post.empty:
        return None

    feat: Dict[str, Any] = {
        'ticker': event.get('ticker'),
        'filingDate': event.get('filingDate'),
        'pre_days': len(pre),
        'post_days': len(post),
    }

    price_at_t0 = post[price_col].iloc[0] if not post.empty else pre[price_col].iloc[-1]
    feat['price_t0'] = float(price_at_t0)
    # Pre momentum & vol
    r_pre = pre[price_col].pct_change().dropna()
    feat['pre_mean_ret'] = float(r_pre.mean()) if not r_pre.empty else 0.0
    feat['pre_vol'] = float(r_pre.std()) if not r_pre.empty else 0.0
    feat['pre_momentum_30d'] = float(pre[price_col].pct_change(30).dropna().iloc[-1]) if len(pre) > 30 else 0.0
    feat['pre_ad_mean_volume'] = float(pre['volume'].dropna().mean()) if 'volume' in pre.columns else 0.0
    feat['pre_max_drawdown'] = float(((pre[price_col].cummax() - pre[price_col]).max()) / (pre[price_col].cummax().max())) if not pre.empty else 0.0

    # -------------------------
    # Additional pre-window technical features (no leakage)
    # -------------------------
    try:
        pre_last_price = float(pre[price_col].iloc[-1]) if not pre.empty else None
    except Exception:
        pre_last_price = None
    tail60 = pre.tail(60)
    # Threshold proximity features
    try:
        feat['pre_days_below_1_60'] = int((tail60[price_col] < 1.0).sum()) if not tail60.empty else 0
    except Exception:
        logger.debug("pre_days_below_1_60 unavailable for %s", event.get('ticker'))
        feat['pre_days_below_1_60'] = 0
    try:
        feat['pre_days_below_050_60'] = int((tail60[price_col] < 0.50).sum()) if not tail60.empty else 0
    except Exception:
        logger.debug("pre_days_below_050_60 unavailable for %s", event.get('ticker'))
        feat['pre_days_below_050_60'] = 0
    # Longest consecutive run price < 1.0
    try:
        s = (pre[price_col] < 1.0).astype(int)
        groups = (s != s.shift()).cumsum()
        feat['pre_consec_below_1'] = int(s.groupby(groups).sum().max() or 0)
    except Exception:
        logger.debug("pre_consec_below_1 unavailable for %s", event.get('ticker'))
        feat['pre_consec_below_1'] = 0
    # Moving averages and ratios at last pre date
    try:
        ma5 = pre[price_col].rolling(5, min_periods=1).mean().iloc[-1]
        feat['pre_price_vs_ma5'] = float(pre_last_price / ma5) if pre_last_price and ma5 else 1.0
    except Exception:
        feat['pre_price_vs_ma5'] = 1.0
    try:
        ma20 = pre[price_col].rolling(20, min_periods=1).mean().iloc[-1]
        feat['pre_price_vs_ma20'] = float(pre_last_price / ma20) if pre_last_price and ma20 else 1.0
    except Exception:
        feat['pre_price_vs_ma20'] = 1.0
    try:
        ma60 = pre[price_col].rolling(60, min_periods=1).mean().iloc[-1]
        feat['pre_price_vs_ma60'] = float(pre_last_price / ma60) if pre_last_price and ma60 else 1.0
    except Exception:
        feat['pre_price_vs_ma60'] = 1.0
    # RSI14 (Wilder)
    try:
        deltas = pre[price_col].diff().dropna()
        gains = deltas.clip(lower=0.0)
        losses = (-deltas).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feat['pre_rsi14'] = float(rsi.iloc[-1]) if not rsi.dropna().empty else 50.0
    except Exception:
        feat['pre_rsi14'] = 50.0
    # Bollinger position (20-day)
    try:
        ma20_full = pre[price_col].rolling(20, min_periods=20).mean()
        sd20_full = pre[price_col].rolling(20, min_periods=20).std(ddof=0)
        upper = ma20_full.iloc[-1] + 2 * sd20_full.iloc[-1] if not ma20_full.dropna().empty and not sd20_full.dropna().empty else None
        lower = ma20_full.iloc[-1] - 2 * sd20_full.iloc[-1] if not ma20_full.dropna().empty and not sd20_full.dropna().empty else None
        rng = (upper - lower) if (upper is not None and lower is not None) else None
        if pre_last_price is not None and rng and rng > 0:
            pos = (pre_last_price - lower) / rng
            feat['pre_bb_pos'] = float(min(1.0, max(0.0, pos)))
        else:
            feat['pre_bb_pos'] = 0.5
    except Exception:
        feat['pre_bb_pos'] = 0.5
    # Dollar liquidity last 30d
    try:
        if 'volume' in pre.columns and pre_last_price is not None:
            t30 = pre.tail(30)
            feat['pre_dollar_vol_30d'] = float(((t30[price_col] * t30['volume']).dropna()).mean()) if not t30.empty else 0.0
        else:
            feat['pre_dollar_vol_30d'] = 0.0
    except Exception:
        feat['pre_dollar_vol_30d'] = 0.0
    # Range placement
    try:
        pmax = float(pre[price_col].max()) if not pre.empty else None
        pmin = float(pre[price_col].min()) if not pre.empty else None
        feat['pre_pct_off_pre_high'] = float((pre_last_price / pmax) - 1.0) if pre_last_price and pmax else 0.0
        feat['pre_pct_above_pre_low'] = float((pre_last_price / pmin) - 1.0) if pre_last_price and pmin else 0.0
    except Exception:
        feat['pre_pct_off_pre_high'] = 0.0
        feat['pre_pct_above_pre_low'] = 0.0
    # Gap activity last 30d
    try:
        if 'open' in pre.columns:
            t30 = pre.tail(30).copy()
            t30['prev_close'] = pre[price_col].shift(1).tail(30)
            gc = (np.abs(t30['open'] - t30['prev_close']) / t30['prev_close']).dropna()
            feat['pre_gap_count_30d'] = int((gc >= 0.05).sum()) if not gc.empty else 0
        else:
            feat['pre_gap_count_30d'] = 0
    except Exception:
        feat['pre_gap_count_30d'] = 0

    trough_price = float(post[price_col].min())
    trough_idx = post[price_col].idxmin()
    feat['trough_price'] = trough_price
    feat['trough_depth_pct'] = float((trough_price - price_at_t0) / price_at_t0)
    feat['time_to_trough_days'] = int((trough_idx - t0).days)

    # ---- Peak labeling on 5d SMA, strictly within post window ----
    time_to_trough = feat.get('time_to_trough_days')
    time_to_peak_days = None
    peak_return_pct = None
    try:
        if price_col and not post.empty:
            # Build post price series and smooth with 5d simple moving average
            post_p = post[price_col].astype(float).copy()
            sma5 = post_p.rolling(5, min_periods=1).mean()

            # Get trough day index; if missing, compute from post_p
            if time_to_trough is None or pd.isna(time_to_trough):
                trough_rel_idx = int(np.nanargmin(post_p.values))
            else:
                trough_rel_idx = int(max(0, min(len(post_p) - 1, int(time_to_trough))))

            # Define search window for peak AFTER trough (≥ trough_rel_idx)
            if trough_rel_idx < len(sma5):
                window = sma5.iloc[trough_rel_idx:]
                if not window.empty:
                    peak_rel_idx = int(np.nanargmax(window.values))
                    time_to_peak_days = int(peak_rel_idx)  # relative to trough
                    # Compute returns using raw prices at those dates
                    buy_price = float(post_p.iloc[trough_rel_idx])
                    sell_price = float(post_p.iloc[min(trough_rel_idx + peak_rel_idx, len(post_p) - 1)])
                    if buy_price > 0:
                        peak_return_pct = (sell_price / buy_price) - 1.0
    except Exception as e:
        logger.debug("peak labeling failed for %s: %s", event.get("ticker"), e)

    feat['time_to_peak_days'] = time_to_peak_days  # int or None
    feat['peak_return_pct'] = peak_return_pct      # float or None

    # Legacy recovery (price returns to t0); keep for reference but will be overridden by new criteria
    rec_mask = post[post[price_col] >= price_at_t0]
    if not rec_mask.empty:
        rec_idx = rec_mask.index[0]
        legacy_days_from_t0 = int((rec_idx - t0).days)
        legacy_days_from_trough = int((rec_idx - trough_idx).days)
    else:
        legacy_days_from_t0 = None
        legacy_days_from_trough = None

    post_after_trough = post.loc[trough_idx:]
    if len(post_after_trough) >= 3:
        x = np.arange(len(post_after_trough))
        y = post_after_trough[price_col].values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        feat['post_trough_slope'] = float(m)
    else:
        feat['post_trough_slope'] = 0.0

    # Persist new classification fields for downstream (non-disruptive)
    feat['reason'] = event.get('reason')
    feat['reason_conf'] = event.get('reason_conf')
    feat['rule'] = (event.get('reason_fields') or {}).get('rule')
    # Optional broader category fields
    if event.get('category') is not None:
        feat['category'] = event.get('category')
    if event.get('cat_conf') is not None:
        feat['cat_conf'] = event.get('cat_conf')

    # Text regain (if discovered during discovery)
    feat['regained_by_text'] = bool(event.get('regained_by_text', False))
    feat['regain_rule'] = event.get('regain_rule')
    feat['regain_date'] = event.get('regain_date')
    feat['regain_accession'] = event.get('regain_accession')
    feat['regain_doc_url'] = event.get('regain_doc_url')

    # Price regain (Nasdaq): close >= threshold for N consecutive trading days
    threshold = globals().get('PRICE_REGAIN_THRESHOLD', 1.0)
    streak_req = int(globals().get('NASDAQ_STREAK_DAYS', 10) or 10)
    use_price_rule = str(event.get('reason') or '').lower().startswith('price') or ('price' in str(event.get('reason') or '').lower())
    feat['regained_by_price'] = False
    regain_price_days_from_t0 = None
    if use_price_rule and not post.empty:
        cond = (post[price_col] >= float(threshold)).astype(int).values.tolist()
        run = 0; best = 0; end_idx = None
        for i, v in enumerate(cond):
            run = run + 1 if v else 0
            if run >= streak_req:
                best = run
                end_idx = i
                break
        if end_idx is not None and best >= streak_req:
            end_ts = post.index[end_idx]
            regain_price_days_from_t0 = int((end_ts - t0).days)
            feat['regained_by_price'] = True

    # Final recovery label: text OR price regain
    regain_any = bool(feat['regained_by_text'] or feat['regained_by_price'])
    feat['recovered_within_post_days'] = 1 if regain_any else 0
    # Choose earliest recovery day across methods
    chosen_days_from_t0 = None
    if feat['regained_by_text'] and feat.get('regain_date'):
        try:
            rd = pd.Timestamp(dt.datetime.strptime(str(feat['regain_date']), "%Y-%m-%d"))
            chosen_days_from_t0 = int((rd - t0).days)
        except Exception:
            pass
    if regain_price_days_from_t0 is not None:
        if chosen_days_from_t0 is None or regain_price_days_from_t0 < chosen_days_from_t0:
            chosen_days_from_t0 = regain_price_days_from_t0
    feat['days_from_t0_to_recovery'] = chosen_days_from_t0
    if chosen_days_from_t0 is not None:
        chosen_idx = t0 + pd.Timedelta(days=chosen_days_from_t0)
        feat['days_from_trough_to_recovery'] = int(((chosen_idx) - trough_idx).days)
    else:
        feat['days_from_trough_to_recovery'] = None
    feat['recovered_method'] = (
        'both' if (feat['regained_by_text'] and feat['regained_by_price']) else
        ('text' if feat['regained_by_text'] else ('price' if feat['regained_by_price'] else 'none'))
    )

    # Regained notice meta (if category explicitly indicates it)
    try:
        feat['is_regained_notice'] = 1 if str(event.get('category') or '').lower() == 'delisting_recovered' else 0
    except Exception:
        feat['is_regained_notice'] = 0

    # Market cap at filing: price_t0 * shares_os (from SEC companyfacts)
    shares_os = None; market_cap = None
    try:
        cik = event.get('cik')
        if cik:
            asof = dt.datetime.strptime(event['filingDate'], "%Y-%m-%d").date()
            shares_os = sec_latest_shares_outstanding(int(cik), asof)
            if shares_os and price_at_t0:
                market_cap = float(shares_os) * float(price_at_t0)
            else:
                logger.warning("market cap: missing shares for %s (%s) as of %s; keeping event",
                               event.get('ticker'), cik, event.get('filingDate'))
    except Exception as e:
        logger.warning("market cap calc failed for %s: %s", event.get('ticker'), e)

    feat['shares_os'] = shares_os
    feat['market_cap'] = market_cap

    return feat

def featurize_events(pre_window_days: int = 180, post_window_days: int = 365, force: bool = False):
    if not EVENTS_FILE.exists():
        logger.error("No events found. Run discover first.")
        return None
    events = json.loads(EVENTS_FILE.read_text())
    features = []
    cap_max = globals().get("CAP_MAX_FILTER", 5e8)
    for ev in tqdm(events, desc="featurize"):
        out = compute_event_features(ev, pre_window_days, post_window_days)
        if out:
            # Enforce cap filter if enabled
            if cap_max and cap_max > 0:
                mc = out.get('market_cap')
                if mc is not None and mc > cap_max:
                    logger.info("skip %s on cap filter: market_cap=%.0f > cap_max=%.0f",
                                out.get('ticker'), mc, cap_max)
                    continue
            features.append(out)
            if len(features) == 1:
                logger.info(
                    "featurize: regained fields present? regained_by_text=%s regained_by_price=%s regain_rule=%s regain_date=%s regain_accession=%s regain_doc_url=%s",
                    'yes' if out.get('regained_by_text') is not None else 'no',
                    'yes' if out.get('regained_by_price') is not None else 'no',
                    'yes' if out.get('regain_rule') is not None else 'no',
                    'yes' if out.get('regain_date') is not None else 'no',
                    'yes' if out.get('regain_accession') is not None else 'no',
                    'yes' if out.get('regain_doc_url') is not None else 'no',
                )
            if len(features) == 1:
                logger.debug("featurize: new pre-window tech columns computed: %s",
                             [c for c in ['pre_rsi14','pre_bb_pos','pre_price_vs_ma20','pre_days_below_1_60','pre_dollar_vol_30d'] if c in out])
    if not features:
        logger.error("No features computed.")
        return None
    df = pd.DataFrame(features)
    df.to_pickle(FEATURES_FILE)
    logger.info("Saved features to %s (%d rows)", FEATURES_FILE, len(df))
    return df

# -------------------------
# Modeling
# -------------------------
def train_models(force: bool = False,
                 train_start: Optional[str] = None,
                 train_end: Optional[str] = None,
                 test_start: Optional[str] = None,
                 test_end: Optional[str] = None):
    if not Path(FEATURES_FILE).exists():
        logger.error("No features found. Run featurize first.")
        return None
    df_all = pd.read_pickle(FEATURES_FILE)
    # Apply optional training date filter by filingDate
    df = df_all
    ts = _parse_date_or_none(train_start)
    te = _parse_date_or_none(train_end)
    tss = _parse_date_or_none(test_start)
    tee = _parse_date_or_none(test_end)
    if "filingDate" in df_all.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_all["filingDate"]):
                df_all["filingDate"] = pd.to_datetime(df_all["filingDate"], errors="coerce")
            df_all["_fdate"] = df_all["filingDate"].dt.date
        except Exception:
            pass
        before_n = len(df_all)
        mask = pd.Series(True, index=df_all.index)
        if ts:
            mask &= (df_all["_fdate"] >= ts)
        if te:
            mask &= (df_all["_fdate"] <= te)
        df = df_all.loc[mask].copy()
        after_n = len(df)
        if ts or te:
            logger.info("train: date filter applied: %s..%s kept %d/%d rows", ts, te, after_n, before_n)
        if tss or tee:
            mask_t = pd.Series(True, index=df_all.index)
            if tss:
                mask_t &= (df_all["_fdate"] >= tss)
            if tee:
                mask_t &= (df_all["_fdate"] <= tee)
            try:
                logger.info("train: test-window rows (for later eval): %d", int(mask_t.sum()))
            except Exception:
                pass
        if "_fdate" in df.columns:
            try:
                df = df.drop(columns=["_fdate"])
            except Exception:
                pass
    else:
        logger.warning("train: filingDate column not found; proceeding without date filter")
    # target present rows only
    df = df.dropna(subset=['recovered_within_post_days'])
    y = df['recovered_within_post_days'].astype(int)

    # --- base numeric features (preserve existing behavior) ---
    base_numeric = [
        'pre_mean_ret','pre_vol','pre_momentum_30d',
        'pre_ad_mean_volume','pre_max_drawdown',
        'trough_depth_pct','time_to_trough_days',
        # Added pre-window technicals (no leakage)
        'pre_days_below_1_60','pre_consec_below_1','pre_days_below_050_60',
        'pre_price_vs_ma5','pre_price_vs_ma20','pre_price_vs_ma60',
        'pre_rsi14','pre_bb_pos','pre_dollar_vol_30d',
        'pre_pct_off_pre_high','pre_pct_above_pre_low','pre_gap_count_30d'
    ]

    # --- add reason_conf and one-hot(reason) ---
    dfX = df.copy()
    dfX['reason_conf'] = pd.to_numeric(dfX.get('reason_conf'), errors='coerce').fillna(0.0)
    dfX['reason'] = dfX.get('reason').fillna('unknown').astype(str)
    reason_dummies = pd.get_dummies(dfX['reason'], prefix='reason')

    X = pd.concat([
        dfX[base_numeric].astype(float).fillna(0.0),
        dfX[['reason_conf']].astype(float),
        reason_dummies
    ], axis=1)
    feature_cols = list(X.columns)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1))
    ])

    # Choose a safe number of splits for small datasets
    try:
        n_splits = max(2, min(3, int(max(2, len(X)) // 3)))
    except Exception:
        n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="roc_auc", n_jobs=1, error_score=np.nan)
        logger.info("Cross-validated ROC-AUC: %.4f Â± %.4f", scores.mean(), scores.std())
    except Exception as e:
        logger.warning("CV failed: %s", e)

    pipeline.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "classifier.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    logger.info("Saved classifier + feature_cols (%d columns)", len(feature_cols))

    # -------------------------
    # Peak-time regressor (predict time_to_peak_days)
    # -------------------------
    def _train_peak_regressor(X_in: pd.DataFrame, y_in: pd.Series):
        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_in, y_in)
        return rf

    try:
        # Align X to rows with valid integer label
        y_peak = df['time_to_peak_days'] if 'time_to_peak_days' in df.columns else None
        if y_peak is not None:
            mask_peak = y_peak.notna()
            if mask_peak.any():
                X_peak = X.loc[mask_peak].copy()
                y_peak = y_peak.loc[mask_peak].astype(float)
                # minimum rows: 50 (or 30 if dataset is small)
                total_rows = int(len(df))
                min_required = 50 if total_rows >= 200 else 30
                if len(y_peak) >= min_required:
                    rf_peak = _train_peak_regressor(X_peak, y_peak)
                    dump(rf_peak, MODELS_DIR / "regressor_peak.joblib")
                    dump(list(X_peak.columns), MODELS_DIR / "feature_cols_peak.joblib")
                    logger.info(
                        "Saved peak regressor with %d rows and %d features",
                        len(y_peak), X_peak.shape[1],
                    )
                else:
                    logger.warning(
                        "Not enough rows to train peak regressor; found %d (min %d)",
                        len(y_peak), min_required,
                    )
            else:
                logger.warning("No non-null labels for time_to_peak_days; skipping peak regressor")
        else:
            logger.info("time_to_peak_days not available in features; skipping peak regressor")
    except Exception as e:
        logger.warning("Peak regressor training failed: %s", e)

    if LIFELINES_AVAILABLE:
        try:
            surv_df = dfX.copy()
            surv_df['duration'] = surv_df['days_from_t0_to_recovery'].fillna(surv_df['post_days']).astype(float)
            surv_df['event'] = surv_df['recovered_within_post_days'].astype(int)
            surv_X = pd.concat([
                surv_df[base_numeric].astype(float).fillna(0.0),
                surv_df[['reason_conf']].astype(float)
            ], axis=1)
            surv_in = pd.concat([surv_X, surv_df[['duration','event']]], axis=1)
            cph = CoxPHFitter()
            cph.fit(surv_in, duration_col='duration', event_col='event')
            joblib.dump(cph, MODELS_DIR / "survival_model.joblib")
            logger.info("Saved survival model (CoxPH)")
        except Exception as e:
            logger.warning("Survival model training failed: %s", e)
    else:
        logger.info("lifelines not installed; skipping survival model.")
    return pipeline

# -------------------------
# Scoring / scanning
# -------------------------
def load_models():
    clf = joblib.load(MODELS_DIR / "classifier.joblib") if (MODELS_DIR / "classifier.joblib").exists() else None
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib") if (MODELS_DIR / "feature_cols.joblib").exists() else None
    surv = joblib.load(MODELS_DIR / "survival_model.joblib") if (MODELS_DIR / "survival_model.joblib").exists() else None
    return clf, feature_cols, surv

def score_candidates(events: List[Dict[str, Any]], clf, feature_cols) -> pd.DataFrame:
    rows = []
    # Optional: load peak regressor if present
    rf_peak = None; peak_cols = None
    try:
        peak_model_path = MODELS_DIR / "regressor_peak.joblib"
        peak_cols_path = MODELS_DIR / "feature_cols_peak.joblib"
        if peak_model_path.exists() and peak_cols_path.exists():
            rf_peak = load(peak_model_path)
            peak_cols = load(peak_cols_path)
        else:
            logger.info("peak regressor not found; skipping predicted_time_to_peak_days")
    except Exception as e:
        logger.info("Failed loading peak regressor: %s", e)
    for ev in tqdm(events, desc="scoring"):
        feat = compute_event_features(ev)
        if not feat:
            continue
        df_row = pd.DataFrame([feat])
        if 'reason_conf' not in df_row.columns:
            df_row['reason_conf'] = 0.0
        df_row['reason_conf'] = pd.to_numeric(df_row['reason_conf'], errors='coerce').fillna(0.0)
        reason_val = str(df_row.get('reason', pd.Series(['unknown'])).iloc[0]) if 'reason' in df_row.columns else 'unknown'
        reason_dummies = pd.get_dummies(pd.Series([reason_val]), prefix='reason')
        # Build a unified feature row (then select per model)
        Xbase = pd.concat([df_row, reason_dummies], axis=1)
        # Classifier features
        X_clf = Xbase.copy()
        for col in feature_cols:
            if col not in X_clf.columns:
                X_clf[col] = 0.0
        X_clf = X_clf[feature_cols].astype(float).fillna(0.0)
        proba = clf.predict_proba(X_clf)
        if proba.shape[1] == 2:
            p1 = proba[0, 1]
        else:
            # Single-class model: map accordingly
            try:
                classes = getattr(clf, 'classes_', None)
                if classes is not None and len(classes) == 1 and classes[0] == 1:
                    p1 = proba[0, 0]
                else:
                    p1 = 0.0
            except Exception:
                p1 = 0.0
        out_row = {**feat, "prob_recovery": float(p1)}
        # Peak-time prediction if model available
        if rf_peak is not None and peak_cols is not None:
            X_peak_like = Xbase.copy()
            for col in peak_cols:
                if col not in X_peak_like.columns:
                    X_peak_like[col] = 0.0
            X_peak_like = X_peak_like[peak_cols].astype(float).fillna(0.0)
            try:
                pred_peak_float = float(rf_peak.predict(X_peak_like)[0])
                pred_peak_int = int(np.rint(pred_peak_float))
                out_row["predicted_time_to_peak_days_float"] = pred_peak_float
                out_row["predicted_time_to_peak_days"] = pred_peak_int
            except Exception:
                # do not fail scoring if regressor errors on a row
                pass
        rows.append(out_row)
    return pd.DataFrame(rows).sort_values("prob_recovery", ascending=False) if rows else pd.DataFrame()

# -------------------------
# Evaluation
# -------------------------
def load_scored(path: Optional[str] = None) -> pd.DataFrame:
    """Load a scored file (CSV/PKL). If path is None, auto-detect the newest scored file in DATA_DIR."""
    cand_paths: List[Path] = []
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = (DATA_DIR / p) if not p.exists() else p
        if not p.exists():
            raise FileNotFoundError(f"Scored file not found: {p}")
        cand_paths = [p]
    else:
        # Common names in our pipeline
        patterns = [
            "scored_events.csv", "scored_events.pkl", "scored.csv", "*.scored.csv", "*.scored.pkl"
        ]
        for pat in patterns:
            cand_paths.extend(sorted((DATA_DIR).glob(pat)))
        # Fallback: any CSV/PKL that looks like scored
        if not cand_paths:
            cand_paths.extend(sorted((DATA_DIR).glob("*.csv")))
            cand_paths.extend(sorted((DATA_DIR).glob("*.pkl")))
        if not cand_paths:
            raise FileNotFoundError("No scored file candidates found in data/")
        # pick newest by mtime
        cand_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        cand_paths = [cand_paths[0]]

    p = cand_paths[0]
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in (".pkl", ".pickle"):
        df = pd.read_pickle(p)
    else:
        raise ValueError(f"Unsupported scored file extension: {p.suffix}")

    # Normalize filingDate
    if 'filingDate' in df.columns:
        try:
            df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
        except Exception:
            pass
    return df


# -------------------------
# Backtest
# -------------------------
def get_cached_prices(ticker: str) -> Optional[pd.DataFrame]:
    """Return cached daily prices for ticker without any network calls.
    Prefers MASTER caches; falls back to concatenating per-span CSVs.
    Ensures a DatetimeIndex and an 'adj_close' column when possible.
    """
    try:
        # Prefer MASTER files (widest coverage)
        masters = sorted(PRICES_DIR.glob(f"{ticker}__*__MASTER.csv"))
        if masters:
            try:
                df = pd.read_csv(masters[-1], parse_dates=["date"], index_col="date")
                if 'adj_close' not in df.columns and 'adjClose' in df.columns:
                    df = df.rename(columns={'adjClose': 'adj_close'})
                return df.sort_index()
            except Exception:
                pass
        # Else gather all per-span slices for this ticker
        spans = sorted(PRICES_DIR.glob(f"{ticker}__*.csv"))
        parts = []
        for p in spans:
            try:
                dfp = pd.read_csv(p, parse_dates=["date"], index_col="date")
                if 'adj_close' not in dfp.columns and 'adjClose' in dfp.columns:
                    dfp = dfp.rename(columns={'adjClose': 'adj_close'})
                parts.append(dfp)
            except Exception:
                continue
        if parts:
            out = pd.concat(parts).sort_index()
            out = out[~out.index.duplicated(keep='last')]
            return out
    except Exception as e:
        logger.info("get_cached_prices failed for %s: %s", ticker, e)
    return None


def build_feature_row_for_models(row: pd.Series, cols: List[str]) -> pd.DataFrame:
    """Construct a 1xN DataFrame aligned to 'cols' using scored row values.
    Fills missing with 0.0; handles reason dummies and reason_conf.
    """
    try:
        reason_val = str(row.get('reason', 'unknown') or 'unknown')
        reason_val = reason_val.strip()
    except Exception:
        reason_val = 'unknown'
    out = {}
    for c in cols:
        if c == 'reason_conf':
            try:
                out[c] = float(pd.to_numeric(row.get('reason_conf'), errors='coerce'))
            except Exception:
                out[c] = 0.0
        elif c.startswith('reason_'):
            out[c] = 1.0 if c == f"reason_{reason_val}" else 0.0
        else:
            try:
                out[c] = float(pd.to_numeric(row.get(c), errors='coerce'))
            except Exception:
                out[c] = 0.0
    return pd.DataFrame([out])


def _nearest_trading_date(px: pd.DataFrame, target_date: pd.Timestamp, col: str, tolerance_days: int) -> Tuple[Optional[pd.Timestamp], Optional[float], str]:
    dates = px.index.normalize()
    # exact
    mask_exact = dates == target_date
    if mask_exact.any():
        d = dates[mask_exact][0]
        try:
            return d, float(px.loc[mask_exact, col].iloc[0]), 'exact'
        except Exception:
            return d, None, 'exact'
    # walk backward then forward up to tolerance calendar days
    for k in range(1, int(tolerance_days) + 1):
        prev_d = target_date - pd.Timedelta(days=k)
        next_d = target_date + pd.Timedelta(days=k)
        mprev = dates == prev_d
        if mprev.any():
            try:
                return prev_d, float(px.loc[mprev, col].iloc[0]), 'prev'
            except Exception:
                return prev_d, None, 'prev'
        mnext = dates == next_d
        if mnext.any():
            try:
                return next_d, float(px.loc[mnext, col].iloc[0]), 'next'
            except Exception:
                return next_d, None, 'next'
    return None, None, 'not_found'


def sheet_name_for_event(ticker: str, filing_date: dt.date, idx: int = 0) -> str:
    base = f"{ticker}{filing_date.strftime('%Y%m%d')}"
    if idx:
        base = f"{base}_{idx}"
    return base[:31] if len(base) > 31 else base


def _safe_int(val, default: int) -> int:
    try:
        if val is None:
            return int(default)
        v = pd.to_numeric([val], errors='coerce')[0]
        if pd.isna(v):
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _xlsx_write(ws, row, col, val, num_fmt=None):
    import numpy as _np
    import pandas as _pd
    fmt = num_fmt
    # numeric case
    if isinstance(val, (int, float, _np.floating)):
        if _np.isfinite(val):
            if fmt is not None:
                ws.write_number(row, col, float(val), fmt)
            else:
                ws.write_number(row, col, float(val))
        else:
            ws.write_blank(row, col, None)
        return
    # non-numeric
    if val is None or (isinstance(val, float) and _np.isnan(val)) or (isinstance(val, _pd._libs.missing.NAType)):
        ws.write_blank(row, col, None)
        return
    try:
        ws.write_string(row, col, str(val))
    except Exception:
        ws.write_blank(row, col, None)


def _write_summary_sheet(wb, df_summary: pd.DataFrame, sheet_to_row_key: dict):
    ws = wb.add_worksheet("Summary")
    # Sanitize infinities
    df_summary = df_summary.replace([np.inf, -np.inf], np.nan)

    headers = list(df_summary.columns) + ["detail_link"]
    # header
    for j, h in enumerate(headers):
        _xlsx_write(ws, 0, j, h)
    # rows + hyperlink
    for i, row in df_summary.reset_index(drop=True).iterrows():
        for j, h in enumerate(df_summary.columns):
            _xlsx_write(ws, i + 1, j, row.get(h))
        key = (
            row.get("ticker"),
            pd.to_datetime(row.get("filingDate"), errors='coerce').date() if pd.notna(row.get("filingDate")) else None,
        )
        sheet = sheet_to_row_key.get(key)
        if sheet:
            ws.write_url(i + 1, len(df_summary.columns), f"internal:'{sheet}'!A1", string="open")
    ws.freeze_panes(1, 0)


def _render_event_chart(px: pd.DataFrame,
                        t0: dt.date,
                        pre_days: int,
                        post_days: int,
                        filing_date: dt.date,
                        buy_date: dt.date,
                        sell_date: dt.date,
                        notices: list,
                        regains: list,
                        title: str) -> Optional[bytes]:
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    try:
        if px is not None and not px.empty:
            series = px["adj_close"] if "adj_close" in px.columns else (px["close"] if "close" in px.columns else None)
            if series is not None:
                ax.plot(series.index, series.values, label="Adj Close" if "adj_close" in px.columns else "Close")
        if isinstance(t0, dt.date):
            ax.axvspan(pd.Timestamp(t0 - dt.timedelta(days=int(pre_days))), pd.Timestamp(t0), alpha=0.08, label="pre window", color="lightgray")
            ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t0 + dt.timedelta(days=int(post_days))), alpha=0.06, label="post window", color="lightgreen")
        if filing_date:
            ax.axvline(pd.Timestamp(filing_date), linestyle="--", label="Filing (T0)", color="blue")
        if buy_date:
            ax.axvline(pd.Timestamp(buy_date), linestyle="-", label="Pred. Trough", color="red")
        if sell_date:
            ax.axvline(pd.Timestamp(sell_date), linestyle="-", label="Pred. Peak", color="green")
        for d, lab in (notices or []):
            try:
                ax.axvline(pd.Timestamp(d), linestyle=":", color="orange", label=lab)
            except Exception:
                pass
        for d, lab in (regains or []):
            try:
                ax.axvline(pd.Timestamp(d), linestyle=":", color="purple", label=lab)
            except Exception:
                pass
        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", ncol=2)
        ax.grid(True, alpha=0.2)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None


def _collect_sec_annotations(events_df: Optional[pd.DataFrame], ticker: str, start_date: dt.date, end_date: dt.date):
    notices, regains = [], []
    if events_df is None or events_df.empty:
        return notices, regains
    df = events_df.copy()
    if "ticker" in df.columns:
        df = df[df["ticker"] == ticker]
    if "filingDate" in df.columns:
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors='coerce').dt.date
        try:
            df = df[(df["filingDate"] >= start_date) & (df["filingDate"] <= end_date)]
        except Exception:
            pass
    for _, r in df.iterrows():
        if str(r.get("reason", "")).strip():
            fd = r.get("filingDate")
            if fd:
                notices.append((fd, "Item 3.01"))
        if bool(r.get("regained_by_text", False)):
            rd = r.get("regain_date")
            try:
                rd = pd.to_datetime(rd, errors='coerce').date() if rd is not None else r.get("filingDate")
            except Exception:
                rd = r.get("filingDate")
            if rd:
                regains.append((rd, "Regained"))
    return notices, regains


def backtest_write_xlsx(df_rep: pd.DataFrame, df_features: pd.DataFrame) -> Optional[Path]:
    if not _XLSX_OK or plt is None:
        logger.info("backtest: xlsxwriter/matplotlib not available; skipping Excel workbook")
        return None
    try:
        # Load scored (for trough/peak fields) and events annotations (optional)
        try:
            df_scored = load_scored(None)
        except Exception:
            df_scored = pd.DataFrame()
        events_df = None
        if EVENTS_FILE.exists():
            try:
                events_df = pd.DataFrame(json.loads(EVENTS_FILE.read_text()))
            except Exception:
                events_df = None

        # Build enrichment columns for Summary
        df_sum = df_rep.copy()
        for d in (df_sum, df_features, df_scored):
            if isinstance(d, pd.DataFrame) and 'filingDate' in d.columns:
                try:
                    d['filingDate'] = pd.to_datetime(d['filingDate'], errors='coerce')
                except Exception:
                    pass
        # Merge scored peak prediction
        if not df_scored.empty:
            cols_sc = [c for c in ['ticker','filingDate','predicted_time_to_peak_days','trough_price','peak_return_pct'] if c in df_scored.columns]
            try:
                df_sum = df_sum.merge(df_scored[cols_sc], on=['ticker','filingDate'], how='left')
            except Exception:
                pass
        # Merge feature labels and post_days if present
        cols_ft = [c for c in ['ticker','filingDate','time_to_trough_days','time_to_peak_days','post_days','pre_days'] if c in df_features.columns]
        try:
            df_sum = df_sum.merge(df_features[cols_ft], on=['ticker','filingDate'], how='left', suffixes=("", "_feat2"))
        except Exception:
            pass

        out_xlsx = DATA_DIR / 'backtest_report.xlsx'
        try:
            if 'is_selected' in df_rep.columns:
                sel = pd.to_numeric(df_rep['is_selected'], errors='coerce').fillna(0)
                sel_count = int(sel.sum())
            else:
                sel_count = int(len(df_rep))
        except Exception:
            sel_count = int(len(df_rep))
        logger.info("backtest: writing Excel workbook to %s events=%d", out_xlsx, sel_count)
        wb = xlsxwriter.Workbook(str(out_xlsx), {'nan_inf_to_errors': True})

        # Common number formats
        price_fmt = wb.add_format({"num_format": "0.0000"})
        prob_fmt  = wb.add_format({"num_format": "0.000"})
        pct_fmt   = wb.add_format({"num_format": "0.00%"})

        # Prepare detail sheets, keep mapping for Summary hyperlinks
        sheet_map = {}
        used_names = set()
        # Iterate over selected rows only
        df_iter = df_sum
        if 'is_selected' in df_iter.columns:
            try:
                df_iter = df_iter[pd.to_numeric(df_iter['is_selected'], errors='coerce').fillna(0).astype(int) == 1]
            except Exception:
                df_iter = df_iter
        for idx, r in df_iter.reset_index(drop=True).iterrows():
            tk = r.get('ticker'); fd = pd.to_datetime(r.get('filingDate'), errors='coerce')
            if tk is None or pd.isna(tk) or fd is None or pd.isna(fd):
                continue
            fdate = fd.date()
            # Determine window
            pre_days = _safe_int(r.get('pre_days'), _safe_int(r.get('pre_days_feat2'), 180))
            post_days = _safe_int(r.get('post_days'), _safe_int(r.get('post_days_feat2'), 120))
            start = fdate - dt.timedelta(days=pre_days)
            end = fdate + dt.timedelta(days=post_days)
            # Load prices and subset
            px_all = get_cached_prices(str(tk))
            px_win = None
            if px_all is not None and not px_all.empty:
                try:
                    px_win = px_all.loc[(px_all.index.date >= start) & (px_all.index.date <= end)].copy()
                except Exception:
                    px_win = px_all
            # Annotations
            notices, regains = _collect_sec_annotations(events_df, str(tk), start, end)
            # Title
            pr = r.get('prob');
            try:
                pr = float(pr) if pr is not None else None
            except Exception:
                pr = None
            title = f"{tk} — Filing {fdate.isoformat()}" + (f" (prob={pr:.3f})" if pr is not None else "")
            # Render chart
            try:
                buy_date = pd.to_datetime(r.get('buy_date'), errors='coerce').date() if r.get('buy_date') else None
            except Exception:
                buy_date = None
            try:
                sell_date = pd.to_datetime(r.get('sell_date'), errors='coerce').date() if r.get('sell_date') else None
            except Exception:
                sell_date = None
            png = _render_event_chart(px_win, fdate, pre_days, post_days, fdate, buy_date, sell_date, notices, regains, title)

            # Unique sheet name
            base = sheet_name_for_event(str(tk), fdate, 0)
            name = base; k = 1
            while name in used_names:
                name = sheet_name_for_event(str(tk), fdate, k)
                k += 1
            used_names.add(name)
            ws = wb.add_worksheet(name)
            # Data table
            def _w(rw, c, key, val, fmt=None):
                _xlsx_write(ws, rw, c, key)
                _xlsx_write(ws, rw, c+1, val, fmt)
            rowp = 0
            # company/accession from events_df
            comp = acc = None
            if events_df is not None and not events_df.empty:
                try:
                    ev_match = events_df[(events_df.get('ticker') == tk) & (pd.to_datetime(events_df.get('filingDate'), errors='coerce').dt.date == fdate)]
                    if not ev_match.empty:
                        comp = ev_match.iloc[0].get('company'); acc = ev_match.iloc[0].get('accession')
                except Exception:
                    pass
            _w(rowp, 0, 'ticker', tk); rowp += 1
            _w(rowp, 0, 'company', comp); rowp += 1
            _w(rowp, 0, 'filingDate', fdate.isoformat()); rowp += 1
            _w(rowp, 0, 'accession', acc); rowp += 1
            _w(rowp, 0, 'prob', pr, prob_fmt); rowp += 1
            _w(rowp, 0, 'predicted_time_to_peak_days', r.get('predicted_time_to_peak_days')); rowp += 1
            _w(rowp, 0, 'time_to_trough_days', r.get('time_to_trough_days')); rowp += 1
            _w(rowp, 0, 'time_to_peak_days', r.get('time_to_peak_days')); rowp += 1
            trp = r.get('trough_price'); prp = r.get('peak_return_pct')
            _w(rowp, 0, 'trough_price', trp, price_fmt); rowp += 1
            peak_price_est = None
            try:
                if (pd.isna(r.get('sell_price')) or r.get('sell_price') is None) and trp is not None and prp is not None:
                    peak_price_est = float(trp) * (1.0 + float(prp))
            except Exception:
                peak_price_est = None
            _w(rowp, 0, 'peak_price_est', peak_price_est, price_fmt); rowp += 1
            _w(rowp, 0, 'buy_date', r.get('buy_date')); rowp += 1
            _w(rowp, 0, 'buy_price', r.get('buy_price'), price_fmt); rowp += 1
            _w(rowp, 0, 'sell_date', r.get('sell_date')); rowp += 1
            _w(rowp, 0, 'sell_price', r.get('sell_price'), price_fmt); rowp += 1
            _w(rowp, 0, 'gain_per_100', r.get('gain_per_100'), price_fmt); rowp += 1
            _w(rowp, 0, 'used_predicted_trough', r.get('used_predicted_trough')); rowp += 1
            _w(rowp, 0, 'used_predicted_peak', r.get('used_predicted_peak')); rowp += 1
            _w(rowp, 0, 'buy_note', r.get('buy_note')); rowp += 1
            _w(rowp, 0, 'sell_note', r.get('sell_note')); rowp += 1
            _w(rowp, 0, 'buy_offset_days', r.get('buy_offset_days')); rowp += 1
            _w(rowp, 0, 'sell_offset_days', r.get('sell_offset_days')); rowp += 1
            _w(rowp, 0, 'buy_offset_source', r.get('buy_offset_source')); rowp += 1
            _w(rowp, 0, 'sell_offset_source', r.get('sell_offset_source')); rowp += 1
            _w(rowp, 0, 'return_pct', r.get('return_pct'), pct_fmt); rowp += 1
            _w(rowp, 0, 'pre_days', pre_days); rowp += 1
            _w(rowp, 0, 'post_days', post_days); rowp += 1
            _w(rowp, 0, 'threshold', r.get('threshold')); rowp += 1
            _w(rowp, 0, 'units', r.get('units')); rowp += 1
            # Notices
            if notices:
                _w(rowp, 0, 'notice_nature', ', '.join(sorted({lab for _, lab in notices}))); rowp += 1
            if regains:
                _w(rowp, 0, 'regain_notice', ', '.join([str(d) for d,_ in regains])); rowp += 1
            # Insert chart below
            if png:
                try:
                    ws.insert_image(16, 0, f"{tk}_{fdate}.png", {'image_data': io.BytesIO(png)})
                except Exception:
                    pass
            logger.info("backtest: sheet %s created", name)
            # Map for Summary hyperlink
            sheet_map[(tk, fdate)] = name

        # Summary sheet
        _write_summary_sheet(wb, df_sum, sheet_map)
        wb.close()
        return out_xlsx
    except Exception as e:
        logger.info("backtest: Excel generation failed: %s", e)
        return None


def backtest_run(df_scored: pd.DataFrame,
                 df_features: pd.DataFrame,
                 start_date: dt.date,
                 end_date: dt.date,
                 threshold: float,
                 units: int,
                 hold_days: int,
                 proba_col: str,
                 use_predicted_trough: bool,
                 tolerance_days: int,
                 pnl_mode: str = "shares") -> pd.DataFrame:
    # Filter by filingDate window
    df = df_scored.copy()
    if 'filingDate' not in df.columns:
        logger.error("backtest: scored file missing 'filingDate'")
        return pd.DataFrame()
    df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
    mask = (df['filingDate'].dt.date >= start_date) & (df['filingDate'].dt.date <= end_date)
    df = df.loc[mask].copy()
    if df.empty:
        logger.info("backtest: no rows in date window %s..%s", start_date, end_date)
        out = pd.DataFrame(columns=[
            'ticker','filingDate','prob','is_selected','buy_date','buy_price',
            'sell_date','sell_price','gain_per_100','recovered_within_post_days',
            'threshold','units','hold_days_fallback','used_predicted_trough','used_predicted_peak'
        ])
        out_path = DATA_DIR / 'backtest_report.csv'
        out.to_csv(out_path, index=False)
        logger.info("backtest: wrote %s rows=%d", out_path, 0)
        return out

    # Merge in feature labels for fallbacks
    df_feat = None
    try:
        keep_cols = ['ticker','filingDate','time_to_trough_days','time_to_peak_days']
        if 'event_id' in df_features.columns:
            keep_cols = ['event_id'] + keep_cols
        df_feat = df_features[keep_cols].copy()
        df_feat['filingDate'] = pd.to_datetime(df_feat['filingDate'], errors='coerce')
        df = df.merge(df_feat, on=[c for c in ['ticker','filingDate'] if c in df_feat.columns], how='left', suffixes=("", "_feat"))
    except Exception as e:
        logger.info("backtest: feature merge skipped: %s", e)

    # Selection by probability
    if proba_col not in df.columns:
        logger.warning("backtest: probability column '%s' not found; defaulting to zeros", proba_col)
        df['prob'] = 0.0
    else:
        df['prob'] = pd.to_numeric(df[proba_col], errors='coerce').fillna(0.0).astype(float)
    df['is_selected'] = (df['prob'] >= float(threshold)).astype(int)

    # Load models if available
    peak_model = None; peak_cols = None
    trough_model = None; trough_cols = None
    try:
        peak_model = load(MODELS_DIR / 'regressor_peak.joblib')
        peak_cols = load(MODELS_DIR / 'feature_cols_peak.joblib')
    except Exception:
        logger.info("backtest: peak regressor not found; will fallback to label or hold-days")
    try:
        if use_predicted_trough:
            trough_model = load(MODELS_DIR / 'regressor_trough.joblib')
            trough_cols = load(MODELS_DIR / 'feature_cols_trough.joblib')
    except Exception:
        logger.info("backtest: trough regressor not found; will fallback to label or t0")

    rows = []
    for _, r in df.iterrows():
        if not int(r.get('is_selected', 0)):
            continue

        ticker = r.get('ticker')
        if not ticker:
            continue
        t0 = pd.Timestamp(r['filingDate']).normalize()

        # BUY timing: predicted trough (if requested + available) else observed label else t0
        buy_offset = 0
        buy_offset_source = 'fallback_t0'
        if use_predicted_trough and trough_model is not None and trough_cols is not None:
            try:
                xtr = build_feature_row_for_models(r, list(trough_cols))
                buy_offset = int(max(0, round(float(trough_model.predict(xtr)[0]))))
                buy_offset_source = 'predicted_trough'
            except Exception:
                buy_offset = 0; buy_offset_source = 'fallback_t0'
        elif not pd.isna(r.get('time_to_trough_days', np.nan)):
            try:
                buy_offset = int(max(0, int(r.get('time_to_trough_days'))))
                buy_offset_source = 'label_trough'
            except Exception:
                buy_offset = 0; buy_offset_source = 'fallback_t0'
        # Calendar-day target; snap to nearest trading day in cache
        buy_target = (t0 + pd.Timedelta(days=int(buy_offset))).normalize()

        # SELL timing: predicted peak days (relative to trough). Fallback to label or hold_days
        sell_offset_rel = None
        sell_offset_source = 'fallback_hold'
        if peak_model is not None and peak_cols is not None:
            try:
                xpk = build_feature_row_for_models(r, list(peak_cols))
                sell_offset_rel = int(max(1, round(float(peak_model.predict(xpk)[0]))))
                sell_offset_source = 'predicted_peak'
            except Exception:
                sell_offset_rel = None
        if sell_offset_rel is None and not pd.isna(r.get('time_to_peak_days', np.nan)):
            try:
                sell_offset_rel = int(max(1, int(r.get('time_to_peak_days'))))
                sell_offset_source = 'label_peak'
            except Exception:
                sell_offset_rel = None
        if sell_offset_rel is None:
            sell_offset_rel = int(max(1, int(hold_days)))
            sell_offset_source = 'fallback_hold'

        sell_target = (buy_target + pd.Timedelta(days=int(sell_offset_rel))).normalize()

        # Price lookup (cached only)
        px = get_cached_prices(str(ticker))
        final_buy_date = buy_target.date(); final_sell_date = sell_target.date()
        buy_price = np.nan; sell_price = np.nan
        buy_note = ''
        sell_note = ''
        if px is not None and not px.empty:
            price_col = 'adj_close' if 'adj_close' in px.columns else ('close' if 'close' in px.columns else None)
            if price_col:
                # BUY snap
                b_d, b_p, b_note = _nearest_trading_date(px, buy_target, price_col, int(tolerance_days))
                if b_d is not None:
                    final_buy_date = b_d.date()
                    buy_price = b_p
                    buy_note = b_note or ''
                    if b_note != 'exact':
                        logger.info("backtest: snapped BUY %s %s from %s to %s (%s)", ticker, r.get('filingDate'), buy_target.date(), final_buy_date, b_note)
                # SELL snap
                s_d, s_p, s_note = _nearest_trading_date(px, sell_target, price_col, int(tolerance_days))
                if s_d is not None:
                    final_sell_date = s_d.date()
                    sell_price = s_p
                    sell_note = s_note or ''
                    if s_note != 'exact':
                        logger.info("backtest: snapped SELL %s %s from %s to %s (%s)", ticker, r.get('filingDate'), sell_target.date(), final_sell_date, s_note)

        gain_per_100 = np.nan
        return_pct = np.nan
        if np.isfinite(buy_price) and np.isfinite(sell_price) and buy_price > 0:
            try:
                if str(pnl_mode).lower() == 'cash':
                    # Interpret 'units' as cash allocation
                    gain_per_100 = float(units) * ((float(sell_price) / float(buy_price)) - 1.0)
                else:
                    # Interpret 'units' as number of shares
                    gain_per_100 = (float(sell_price) - float(buy_price)) * int(units)
            except Exception:
                if str(pnl_mode).lower() == 'cash':
                    gain_per_100 = float(units) * ((float(sell_price) / float(buy_price)) - 1.0)
                else:
                    gain_per_100 = (float(sell_price) - float(buy_price)) * 100
            try:
                return_pct = (sell_price / buy_price) - 1.0
            except Exception:
                return_pct = np.nan

        rows.append({
            'ticker': str(ticker),
            'filingDate': t0.date().isoformat(),
            'prob': float(r.get('prob', 0.0)),
            'is_selected': int(r.get('is_selected', 0)),
            'buy_date': str(final_buy_date),
            'buy_price': buy_price,
            'sell_date': str(final_sell_date),
            'sell_price': sell_price,
            'gain_per_100': gain_per_100,
            'return_pct': return_pct,
            'recovered_within_post_days': int(r.get('recovered_within_post_days', 0) or 0),
            'threshold': float(threshold),
            'units': int(units),
            'hold_days_fallback': int(hold_days),
            'pnl_mode': str(pnl_mode),
            'used_predicted_trough': bool(use_predicted_trough and (trough_model is not None)),
            'used_predicted_peak': bool(peak_model is not None),
            'buy_note': buy_note,
            'sell_note': sell_note,
            'buy_offset_days': int(buy_offset),
            'sell_offset_days': int(sell_offset_rel),
            'buy_offset_source': buy_offset_source,
            'sell_offset_source': sell_offset_source,
        })

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / 'backtest_report.csv'
    out.to_csv(out_path, index=False)
    logger.info("backtest: wrote %s rows=%d", out_path, len(out))
    return out


def evaluate_scored(df: pd.DataFrame,
                    start: dt.date,
                    end: dt.date,
                    proba_col: str = "prob_recovery",
                    prec_at: int = 50,
                    prec_target: float = 0.70) -> pd.DataFrame:
    """Compute evaluation metrics over [start, end] based on filingDate.
    Returns a one-row DataFrame and saves CSV to data/eval_report.csv.
    """
    if 'filingDate' not in df.columns:
        logger.error("evaluate: filingDate column not found in scored file")
        return pd.DataFrame()
    if 'recovered_within_post_days' not in df.columns:
        logger.error("evaluate: recovered_within_post_days column not found in scored file")
        return pd.DataFrame()
    if proba_col not in df.columns:
        logger.error("evaluate: probability column '%s' not found in scored file", proba_col)
        return pd.DataFrame()

    # Coerce types
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['filingDate']):
        df['filingDate'] = pd.to_datetime(df['filingDate'], errors='coerce')
    df = df.dropna(subset=['filingDate', 'recovered_within_post_days', proba_col])

    # Filter window
    mask = (df['filingDate'].dt.date >= start) & (df['filingDate'].dt.date <= end)
    dft = df.loc[mask].copy()
    n = len(dft)
    if n == 0:
        logger.warning("evaluate: no rows in test window %s..%s", start, end)
        return pd.DataFrame()

    y = dft['recovered_within_post_days'].astype(int).values
    p = dft[proba_col].astype(float).values
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())

    # Metrics
    roc = float('nan'); ap = float('nan')
    if n_pos > 0 and n_neg > 0:
        try:
            roc = float(roc_auc_score(y, p))
        except Exception:
            pass
        try:
            ap = float(average_precision_score(y, p))
        except Exception:
            pass
    try:
        brier = float(brier_score_loss(y, p))
    except Exception:
        brier = float('nan')

    # Precision@K
    precK = float('nan')
    if n > 0:
        K = max(1, min(int(prec_at), n))
        order = dft.sort_values(proba_col, ascending=False).head(K)
        try:
            precK = float(order['recovered_within_post_days'].astype(int).mean())
        except Exception:
            pass

    # Best F1 threshold via precision-recall curve
    best_f1 = float('nan'); best_thr = float('nan')
    thr_prec = float('nan'); thr_rec = float('nan')
    try:
        prec, rec, thr = precision_recall_curve(y, p)
        # Map thresholds to prec/rec at index i -> threshold[i-1]
        if len(thr) > 0:
            f1s = []
            for i in range(1, len(prec)):
                f1 = 2 * prec[i] * rec[i] / (prec[i] + rec[i] + 1e-9)
                f1s.append((f1, thr[i-1], prec[i], rec[i]))
            f1s.sort(key=lambda x: x[0], reverse=True)
            best_f1, best_thr, _, _ = f1s[0]
    except Exception:
        pass

    # Threshold to achieve target precision
    tgt_thr = float('nan'); tgt_prec = float('nan'); tgt_rec = float('nan')
    TP = FP = FN = TN = None
    try:
        prec, rec, thr = precision_recall_curve(y, p)
        if len(thr) > 0:
            candidates = []
            for i in range(1, len(prec)):
                if prec[i] >= prec_target:
                    candidates.append((thr[i-1], prec[i], rec[i]))
            if candidates:
                candidates.sort(key=lambda x: x[0])  # smallest threshold achieving target precision
                tgt_thr, tgt_prec, tgt_rec = candidates[0]
                # Confusion counts at tgt_thr
                pred = (p >= tgt_thr).astype(int)
                TP = int(((pred == 1) & (y == 1)).sum())
                FP = int(((pred == 1) & (y == 0)).sum())
                FN = int(((pred == 0) & (y == 1)).sum())
                TN = int(((pred == 0) & (y == 0)).sum())
    except Exception:
        pass

    logger.info(
        "evaluate %s..%s: N=%d pos=%d neg=%d | ROC-AUC=%.4f AP=%.4f Brier=%.4f | P@%d=%.4f | bestF1=%.4f@thr=%.4f | tgtPrec=%.2f thr=%.4f prec=%.4f rec=%.4f TP=%s FP=%s FN=%s TN=%s",
        start, end, n, n_pos, n_neg,
        roc if not pd.isna(roc) else float('nan'),
        ap if not pd.isna(ap) else float('nan'),
        brier if not pd.isna(brier) else float('nan'),
        int(min(prec_at, n)),
        precK if not pd.isna(precK) else float('nan'),
        best_f1 if not pd.isna(best_f1) else float('nan'),
        best_thr if not pd.isna(best_thr) else float('nan'),
        float(prec_target),
        tgt_thr if not pd.isna(tgt_thr) else float('nan'),
        tgt_prec if not pd.isna(tgt_prec) else float('nan'),
        tgt_rec if not pd.isna(tgt_rec) else float('nan'),
        TP, FP, FN, TN,
    )

    report = pd.DataFrame([{
        'start': start, 'end': end,
        'n': n, 'n_pos': n_pos, 'n_neg': n_neg,
        'roc_auc': roc, 'avg_precision': ap, 'brier': brier,
        'prec_at': int(min(prec_at, n)), 'precision_at_k': precK,
        'best_f1': best_f1, 'best_f1_threshold': best_thr,
        'prec_target': float(prec_target), 'thr_prec_target': tgt_thr,
        'precision_at_target': tgt_prec, 'recall_at_target': tgt_rec,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
    }])
    try:
        out = DATA_DIR / 'eval_report.csv'
        report.to_csv(out, index=False)
        logger.info("evaluate: wrote %s", out)
    except Exception:
        pass
    return report

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Delisting recovery pipeline")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("discover", help="Discover delisting notices via SEC search API")
    d.add_argument("--years", type=int, default=5)
    d.add_argument("--size", type=int, default=200, help="Search page size (50..500)")
    d.add_argument("--throttle", type=float, default=0.1, help="Sleep between requests (sec)")
    d.add_argument("--strict", action="store_true", help="Strict validation (drop docs mentioning 'not applicable')")
    d.add_argument("--max-pages", type=int, default=None, help="Limit number of pages for testing")
    d.add_argument("--force", action="store_true", help="Ignore cached events.json")
    d.add_argument("--log", type=str, default="INFO", help="Log level: DEBUG, INFO, WARNING, ERROR")
    d.add_argument("--no-proxy", action="store_true", help="Force no proxies (ignore proxies.txt and skip proxy attempts)")
    d.add_argument("--price-only", action="store_true", help="Keep only price/market-value related Item 3.01 events")
    d.add_argument("--detect-regain-text", action="store_true", default=True, help="Scan post-window 8-Ks to detect text-based regain of compliance (default: on)")
    d.add_argument("--regain-post-days", type=int, default=365, help="Post window (days) for regain text detection")

    f = sub.add_parser("fetch_prices", help="Fetch Tiingo prices for discovered events")
    f.add_argument("--pre-days", type=int, default=180)
    f.add_argument("--post-days", type=int, default=365)
    f.add_argument("--force", action="store_true")
    f.add_argument("--log", type=str, default="INFO")
    f.add_argument("--max-tickers", type=int, default=None,
                   help="Limit number of distinct tickers for testing.")
    f.add_argument("--rate-sleep", type=float, default=72.0,
                   help="Seconds to sleep between Tiingo API calls (default=72 (50/h); helps free-tier rate limits).")
    f.add_argument("--tiingo-max-requests", type=int, default=-1,
                   help="Optional cap on total Tiingo requests for this run (default: no cap).")

    feat = sub.add_parser("featurize", help="Compute features (pre-window technical; no post leakage)")
    feat.add_argument("--pre-days", type=int, default=180)
    feat.add_argument("--post-days", type=int, default=365)
    feat.add_argument("--log", type=str, default="INFO")
    feat.add_argument("--cap-max", type=float, default=5e8,
                      help="Max market cap (USD) to keep an event. Default=5e8. Use -1 to disable filter.")
    feat.add_argument("--price-threshold", type=float, default=1.0,
                      help="Close price threshold for price-based regain (Nasdaq) [default=1.0]")
    feat.add_argument("--nasdaq-streak", type=int, default=10,
                      help="Consecutive trading days at/above threshold for regain [default=10]")

    train = sub.add_parser(
        "train",
        help="Train ML models",
        epilog=(
            "You can restrict training to a time range with --train-start/--train-end.\n"
            "Example: --train-end 2025-06-30 to train on Q1–Q2 2025 only; evaluate Q3 separately."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    train.add_argument("--force", action="store_true")
    train.add_argument("--log", type=str, default="INFO")
    train.add_argument("--train-start", type=str, default=None,
                       help="Start date (YYYY-MM-DD) for training rows, inclusive")
    train.add_argument("--train-end", type=str, default=None,
                       help="End date (YYYY-MM-DD) for training rows, inclusive")
    train.add_argument("--test-start", type=str, default=None,
                       help="Start date (YYYY-MM-DD) for held-out reporting (no effect on fitting)")
    train.add_argument("--test-end", type=str, default=None,
                       help="End date (YYYY-MM-DD) for held-out reporting (no effect on fitting)")

    score = sub.add_parser("score_universe", help="Score the cached events or all tickers")
    score.add_argument("--mode", choices=["cached_events", "predict_all"], default="cached_events")
    score.add_argument("--log", type=str, default="INFO")

    # Evaluate
    ev = sub.add_parser(
        "evaluate",
        help="Evaluate scored predictions over a test window",
        epilog=(
            "Example:\n"
            "  python delist_recovery_pipeline.py evaluate --test-start 2025-07-01 --test-end 2025-09-30\n"
            "Optional: --scored-path data/scored_events.csv --prec-at 100 --prec-target 0.8"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ev.add_argument("--test-start", type=str, required=True, help="YYYY-MM-DD inclusive")
    ev.add_argument("--test-end", type=str, required=True, help="YYYY-MM-DD inclusive")
    ev.add_argument("--scored-path", type=str, default=None, help="Path to scored file; if None, auto-detect in data/")
    ev.add_argument("--proba-col", type=str, default="prob_recovery", help="Probability column name")
    ev.add_argument("--prec-at", type=int, default=50, help="K for Precision@K (uses min(K, N))")
    ev.add_argument("--prec-target", type=float, default=0.70, help="Target precision to find threshold for")

    # Backtest
    bt = sub.add_parser(
        "backtest",
        help="Simulate P&L per event and write CSV",
        epilog=(
            "Example:\n"
            "  python stocks\\delist_recovery_pipeline.py backtest --test-start 2025-07-01 --test-end 2025-09-30 --threshold 0.6 --units 100 --hold-days 20\n"
            "Loads scored events, applies threshold, and simulates BUY (t0+trough) to SELL (trough+peak) using cached prices."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    bt.add_argument("--test-start", type=str, required=True, help="YYYY-MM-DD inclusive")
    bt.add_argument("--test-end", type=str, required=True, help="YYYY-MM-DD inclusive")
    bt.add_argument("--threshold", type=float, default=0.6, help="Probability cutoff to select trades")
    bt.add_argument("--units", type=int, default=100, help="Units to buy per trade (for P&L calculation)")
    bt.add_argument("--hold-days", type=int, default=20, help="Fallback hold days if no predicted/label peak available")
    bt.add_argument("--tolerance-days", type=int, default=5, help="Nearest trading day snap tolerance (+/- days)")
    bt.add_argument("--proba-col", type=str, default="prob_recovery", help="Probability column name in scored file")
    bt.add_argument("--pnl-mode", choices=["shares","cash"], default="shares", help="P&L mode: 'shares' uses (sell−buy)*units; 'cash' uses units*((sell/buy)−1)")
    bt.add_argument("--scored-path", type=str, default=None, help="Path to scored file; auto-detect if None")
    bt.add_argument("--use-predicted-trough", action="store_true", help="Use predicted trough model if available; else fallback")
    bt.add_argument("--log", type=str, default="INFO")
    bt.add_argument("--xlsx", action="store_true", help="Also write Excel workbook with per-event charts")

    sp = sub.add_parser("show_proxies", help="Show loaded proxies and exit")
    sp.add_argument("--log", type=str, default="INFO")

    # New: quarterly index discovery (robust, no efts usage)
    di = sub.add_parser("discover_index", help="Discover via quarterly full-index master.idx")
    di.add_argument("--years", type=int, default=5)
    di.add_argument("--throttle", type=float, default=0.1)
    di.add_argument("--strict", action="store_true")
    di.add_argument("--force", action="store_true")
    di.add_argument("--log", type=str, default="INFO")
    di.add_argument("--log-every-event", type=int, default=50, help="Persist events/progress every N items processed/validated")
    di.add_argument("--price-only", action="store_true", help="Keep only price/market-value related Item 3.01 events")
    di.add_argument("--threads", type=int, default=1, help="Parallel threads for document fetch (default=1; only with proxies)")
    di.add_argument("--detect-regain-text", action="store_true", default=True, help="Scan post-window 8-Ks to detect text-based regain of compliance (default: on)")
    di.add_argument("--regain-post-days", type=int, default=365, help="Post window (days) for regain text detection")

    return p.parse_args()

def main():
    args = parse_args()
    setup_logging(getattr(args, "log", "INFO"))
    load_proxies()
    global NO_PROXY
    NO_PROXY = getattr(args, "no_proxy", False)
    if NO_PROXY:
        PROXIES.clear()
        logger.info("NO_PROXY set: proxy list cleared; all requests will be direct.")
    logger.info("Loaded %d proxies", len(PROXIES))

    if args.cmd == "discover":
        discover_notices(
            years=args.years,
            size=args.size,
            throttle=args.throttle,
            force=args.force,
            strict=args.strict,
            max_pages=args.max_pages,
            price_only=getattr(args, "price_only", False),
            detect_regain_text=getattr(args, "detect_regain_text", True),
            regain_post_days=getattr(args, "regain_post_days", 365),
        )
    elif args.cmd == "discover_index":
        discover_via_index(
            years=args.years,
            throttle=args.throttle,
            force=args.force,
            strict=args.strict,
            price_only=getattr(args, "price_only", False),
            log_every_event=getattr(args, "log_every_event", 50),
            detect_regain_text=getattr(args, "detect_regain_text", True),
            regain_post_days=getattr(args, "regain_post_days", 365),
        )
    elif args.cmd == "fetch_prices":
        if not EVENTS_FILE.exists():
            logger.error("No events.json found; run discover first.")
            sys.exit(1)
        events = json.loads(EVENTS_FILE.read_text())
        # Optional: limit number of distinct tickers for testing
        if getattr(args, "max_tickers", None):
            seen = set()
            limited = []
            for ev in events:
                t = ev.get("ticker")
                if not t:
                    continue
                if len(seen) >= args.max_tickers and t not in seen:
                    continue
                seen.add(t)
                limited.append(ev)
            events = limited
            logger.info("Limiting to first %d tickers; %d events remain.", args.max_tickers, len(events))

        # Optional per-run Tiingo cap
        try:
            globals()["TIINGO_MAX_REQUESTS"] = int(getattr(args, "tiingo_max_requests", -1) or -1)
        except Exception:
            globals()["TIINGO_MAX_REQUESTS"] = -1

        fetch_prices_for_events(
            events,
            pre_days=args.pre_days, post_days=args.post_days,
            force=args.force,
            rate_sleep=getattr(args, "rate_sleep", 0.0)
        )
    elif args.cmd == "featurize":
        # Apply market-cap filter from CLI (default 5e8; -1 disables)
        global CAP_MAX_FILTER
        try:
            CAP_MAX_FILTER = getattr(args, "cap_max", 5e8)
        except Exception:
            CAP_MAX_FILTER = 5e8
        # Apply price-regain parameters
        global PRICE_REGAIN_THRESHOLD, NASDAQ_STREAK_DAYS
        PRICE_REGAIN_THRESHOLD = getattr(args, "price_threshold", 1.0)
        NASDAQ_STREAK_DAYS = getattr(args, "nasdaq_streak", 10)
        featurize_events(pre_window_days=args.pre_days, post_window_days=args.post_days)
    elif args.cmd == "train":
        train_models(
            force=args.force,
            train_start=getattr(args, "train_start", None),
            train_end=getattr(args, "train_end", None),
            test_start=getattr(args, "test_start", None),
            test_end=getattr(args, "test_end", None),
        )
    elif args.cmd == "score_universe":
        clf, feature_cols, _ = load_models()
        if clf is None:
            logger.error("No classifier found; run train first.")
            sys.exit(1)
        if args.mode == "cached_events":
            events = json.loads(EVENTS_FILE.read_text())
            df = score_candidates(events, clf, feature_cols)
            out = DATA_DIR / "scored_events.csv"
            df.to_csv(out, index=False)
            logger.info("Saved scored events to %s", out)
        else:
            logger.info("predict_all mode not implemented in this revision.")
    elif args.cmd == "evaluate":
        try:
            start = pd.to_datetime(getattr(args, "test_start"), errors='coerce').date()
            end = pd.to_datetime(getattr(args, "test_end"), errors='coerce').date()
        except Exception:
            logger.error("Invalid --test-start/--test-end provided.")
            sys.exit(1)
        df = load_scored(getattr(args, "scored_path", None))
        evaluate_scored(
            df,
            start,
            end,
            getattr(args, "proba_col", "prob_recovery"),
            int(getattr(args, "prec_at", 50) or 50),
            float(getattr(args, "prec_target", 0.70) or 0.70),
        )
    elif args.cmd == "backtest":
        try:
            start = pd.to_datetime(getattr(args, "test_start"), errors='coerce').date()
            end = pd.to_datetime(getattr(args, "test_end"), errors='coerce').date()
        except Exception:
            logger.error("Invalid --test-start/--test-end provided.")
            sys.exit(1)
        scored = load_scored(getattr(args, "scored_path", None))
        if not Path(FEATURES_FILE).exists():
            logger.error("No features found at %s; run featurize first.", FEATURES_FILE)
            sys.exit(1)
        feats = pd.read_pickle(FEATURES_FILE)
        df_bt = backtest_run(
            scored, feats,
            start, end,
            float(getattr(args, "threshold", 0.6) or 0.6),
            int(getattr(args, "units", 100) or 100),
            int(getattr(args, "hold_days", 20) or 20),
            str(getattr(args, "proba_col", "prob_recovery") or "prob_recovery"),
            bool(getattr(args, "use_predicted_trough", False)),
            int(getattr(args, "tolerance_days", 5) or 5),
        )
        if bool(getattr(args, "xlsx", False)):
            try:
                backtest_write_xlsx(df_bt, feats)
            except Exception as e:
                logger.info("backtest: xlsx generation error: %s", e)
    elif args.cmd == "show_proxies":
        for i, pxy in enumerate(PROXIES, 1):
            logger.info("[%02d] %s", i, pxy.get("_display", "direct"))
    else:
        print("No command provided. Use --help.")

if __name__ == "__main__":
    main()
