#!/usr/bin/env python3
"""
Delisting Recovery Pipeline — FULL SCRIPT (updated)

What's included
---------------
1) Discovery uses SEC full-text search API (no ticker loop) to find 8-K filings.
2) ITEM 3.01 validation: fetch primary 8-K document and confirm real section header
   with delisting context (e.g., "Notice of Delisting", "Failure to Satisfy...").
3) Proxy rotation via proxies.txt (Webshare format: IP:port:login:password).
4) Verbose tracing: proxy used (IP:port), URLs, HTTP status, bytes, and result per filing.
5) Full pipeline retained: Tiingo fetch, featurize, train, score.

Environment
-----------
- Set TIINGO_API_KEY in your environment for Tiingo price downloads.
- Optional: create proxies.txt for Webshare proxies next to this script.

"""

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

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

# -------------------------
# Config & constants
# -------------------------
DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices"
MODELS_DIR = DATA_DIR / "models"
EVENTS_FILE = DATA_DIR / "events.json"
TICKERS_FILE = DATA_DIR / "tickers_nasdaq.json"
FEATURES_FILE = DATA_DIR / "features.pkl"
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

logger = logging.getLogger("delist_pipeline")

# -------------------------
# Proxy rotation (Webshare)
# -------------------------
PROXIES_FILE = Path("proxies.txt")
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
            logger.info("  → HTTP %s (%d bytes)", r.status_code, len(r.content))
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
            logger.info("  → HTTP %s (%d bytes)", r.status_code, len(r.content))
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
    # Direct first
    for attempt in range(1, retries + 1):
        logger.info("GET %s [attempt %d/%d] proxy=%s → %s", desc, attempt, retries, "direct", url)
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
            logger.info("  ← HTTP %s (%d bytes)", r.status_code, len(r.content))
            if r.status_code == 200:
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
        logger.info("GET %s [proxy attempt %d/%d] proxy=%s → %s", desc, attempt, retries, disp, url)
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
            logger.info("  ← HTTP %s (%d bytes)", r.status_code, len(r.content))
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
    logger.info("ATOM page %d → %d entries", page, len(entries))
    return entries

# -------------------------
# SEC helpers
# -------------------------
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

def extract_primary_doc_url(index_html: str, index_url: str) -> Optional[str]:
    soup = BeautifulSoup(index_html, "html.parser")
    # Try the standard Documents table first
    table = soup.find("table", {"class": "tableFile", "summary": "Document Format Files"})
    if table:
        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header
            cols = row.find_all("td")
            if not cols or len(cols) < 3:
                continue
            a = cols[2].find("a")
            href = a.get("href") if a else None
            if href and href.lower().endswith((".htm", ".html")):
                return href if href.startswith("http") else ("https://www.sec.gov" + href)
    # Fallback: first HTML link not containing 'index'
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.lower().endswith((".htm", ".html")) and "index" not in href.lower():
            return href if href.startswith("http") else ("https://www.sec.gov" + href)
    return None

def validate_item_301(doc_html: str, strict: bool = True) -> bool:
    # Parse HTML to text (strip tags)
    text = BeautifulSoup(doc_html, "html.parser").get_text(" ", strip=True)
    low = text.lower()
    # Must contain ITEM 3.01 header
    if not ITEM_HEADER_RE.search(low):
        return False
    # Must *not* be "not applicable" (often boilerplate)
    if NOT_APPL_RE.search(low):
        # Could still be valid if context phrases appear; keep strict behaviour
        if strict:
            return False
    # Must contain one context phrase somewhere in doc (to reduce false positives)
    return CONTEXT_RE.search(low) is not None

# -------------------------
# Submission .txt prefilter (fast drop of non-price events)
# -------------------------
# Strong positive signals (price / market-cap / float / equity deficiencies)
POS_PRICE_PATTERNS = [
    re.compile(r"nasdaq\s+(listing\s+)?rule\s*5550\(a\)\(2\)", re.I),
    re.compile(r"nyse(\s+american)?\s+section\s*802\.01c", re.I),
    re.compile(r"minimum\s+bid\s+price", re.I),
    re.compile(r"bid\s+price[^\n]{0,100}\$?1(\.00)?", re.I),
    re.compile(r"30\s+consecutive\s+(trading|business)\s+days", re.I),
    re.compile(r"180[- ]calendar[- ]day(?:s)?\s+(?:compliance|period)", re.I),
]
POS_MV_PATTERNS = [
    re.compile(r"market\s+value\s+of\s+(publicly\s+held\s+shares|listed\s+securities|public\s+float)", re.I),
    re.compile(r"public\s+float", re.I),
    re.compile(r"market\s+capitalization|market\s+cap", re.I),
    re.compile(r"stockholders'?\s+equity", re.I),
    re.compile(r"nasdaq\s+(listing\s+)?rule\s*5550\(b\)", re.I),
    re.compile(r"nyse(\s+american)?\s+section\s*802\.01b", re.I),
]

# Definitive negatives (mergers/voluntary delisting etc.)
NEG_MERGER_PATTERNS = [
    re.compile(r"in\s+connection\s+with\s+the\s+merger", re.I),
    re.compile(r"\bmerger\b|acquisition|business\s+combination|consummation\s+of\s+the\s+merger", re.I),
    re.compile(r"voluntary\s+delist(ing)?|voluntary\s+deregistration", re.I),
    re.compile(r"form\s+25[^\n]{0,80}in\s+connection\s+with\s+the\s+merger", re.I),
    re.compile(r"tender\s+offer|going\s+private|liquidation|dissolution|redeem(?:ing)?\s+all\s+shares", re.I),
]
# Other non-price rules often not tied to price/market value
NEG_OTHER_RULE_PATTERNS = [
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

    pos_hits = any(p.search(low) for p in POS_PRICE_PATTERNS) or any(p.search(low) for p in POS_MV_PATTERNS)
    for p in POS_PRICE_PATTERNS + POS_MV_PATTERNS:
        m = p.search(low)
        if m:
            info["matched"].append(m.group(0)[:80])

    neg_def = False
    for p in NEG_MERGER_PATTERNS + NEG_OTHER_RULE_PATTERNS:
        m = p.search(low)
        if m:
            info["negatives"].append(m.group(0)[:80])
            # Mark definitive negatives (merger/voluntary) as blockers
            if p in NEG_MERGER_PATTERNS:
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

def classify_item301(section_text: str) -> Dict[str, Any]:
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
    for p in POS_PRICE_PATTERNS:
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
def discover_notices(years: int = 5, size: int = 200, throttle: float = 0.1, force: bool = False, strict: bool = True, max_pages: Optional[int] = None, price_only: bool = False) -> List[Dict[str, Any]]:
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

                doc_url = extract_primary_doc_url(index_html, index_url)
                logger.info("  primary doc url: %s", doc_url or "NONE")
                if not doc_url:
                    continue
                r_doc, doc_html, _ = http_get(doc_url, f"doc for {accession}")
                if r_doc is None or r_doc.status_code != 200:
                    logger.info("  doc fetch failed; skipping")
                    continue

                is_item = validate_item_301(doc_html, strict=strict)
                logger.info("  ITEM 3.01 validation: %s", "FOUND" if is_item else "NOT FOUND")
                if not is_item:
                    polite_sleep(throttle); continue

                section = extract_item_301_section(doc_html)
                cls = classify_item301(section)
                if price_only and cls.get("label") not in ("price_bid_1", "mv_equity_float"):
                    polite_sleep(throttle); continue

                if not ticker:
                    tk, _ = cik_to_ticker_name(cik)
                    ticker = tk

                key = (cik, accession)
                if key not in results:
                    results[key] = {
                        "cik": cik, "ticker": ticker, "company": company, "filingDate": filed_at,
                        "accession": accession, "filing_index_url": index_url, "filing_doc_url": doc_url,
                        "classification": cls,
                    }
                    total_validated += 1
                    if total_validated % 25 == 0:
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
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

                doc_url = extract_primary_doc_url(index_html, index_url)
                logger.info("  primary doc url: %s", doc_url or "NONE")
                if not doc_url:
                    continue
                r_doc, doc_html, _ = http_get(doc_url, f"doc (ATOM)")
                if r_doc is None or r_doc.status_code != 200:
                    logger.info("  doc fetch failed; skipping")
                    continue

                is_item = validate_item_301(doc_html, strict=strict)
                logger.info("  ITEM 3.01 validation: %s", "FOUND" if is_item else "NOT FOUND")
                if not is_item:
                    polite_sleep(throttle); continue

                section = extract_item_301_section(doc_html)
                cls = classify_item301(section)
                if price_only and cls.get("label") not in ("price_bid_1", "mv_equity_float"):
                    polite_sleep(throttle); continue

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

def discover_via_index(years: int = 5, throttle: float = 0.1, force: bool = False, strict: bool = True, price_only: bool = False) -> List[Dict[str, Any]]:
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

            for r in tqdm(candidates, desc=f"{year}Q{qtr} 8-K", unit="filing", leave=False):
                filed_at = r.get("date")
                accession, index_url = _derive_accession_and_urls(r)
                if not accession or not index_url:
                    continue
                try:
                    cik = int(r.get("cik")) if r.get("cik") else None
                except Exception:
                    cik = None
                # Prefilter using the submission text file to avoid unnecessary HTML fetches
                sub_url = f"https://www.sec.gov/Archives/{r.get('filename')}"
                r_sub, sub_txt, _ = http_get(sub_url, f"submission txt for {accession}")
                if r_sub is None or r_sub.status_code != 200 or not sub_txt:
                    polite_sleep(throttle)
                    continue
                keep, pf_info = prefilter_submission_text(sub_txt)
                if not keep:
                    polite_sleep(throttle)
                    continue

                r_index, index_html, _ = http_get(index_url, f"index for {accession}")
                if r_index is None or r_index.status_code != 200:
                    continue
                doc_url = extract_primary_doc_url(index_html, index_url)
                if not doc_url:
                    continue
                r_doc, doc_html, _ = http_get(doc_url, f"doc for {accession}")
                if r_doc is None or r_doc.status_code != 200:
                    continue
                is_item = validate_item_301(doc_html, strict=strict)
                if not is_item:
                    polite_sleep(throttle)
                    continue

                # Post-filter classification on the actual Item 3.01 section
                section = extract_item_301_section(doc_html)
                cls = classify_item301(section)
                if price_only and cls.get("label") not in ("price_bid_1", "mv_equity_float"):
                    polite_sleep(throttle)
                    continue

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
                        "prefilter": pf_info,
                        "classification": cls,
                    }
                    total_validated += 1
                    if total_validated % 25 == 0:
                        json.dump(list(results.values()), open(EVENTS_FILE, "w"), indent=2, default=str)
                        logger.info("Persisted %d validated events so far...", total_validated)
                polite_sleep(throttle)

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
            return df
        except Exception:
            pass

    url = f"{TIINGO_BASE}/tiingo/daily/{ticker}/prices"
    params = {"startDate": start_date, "endDate": end_date}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
    logger.info("GET Tiingo %s %s→%s", ticker, start_date, end_date)
    r = requests.get(url, headers=headers, params=params, timeout=30)
    logger.info(" ← %s (%d bytes)", r.status_code, len(r.content))
    if r.status_code != 200:
        logger.error("Tiingo fetch failed for %s: %s %s", ticker, r.status_code, r.text[:200])
        return None
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
    return df

def fetch_prices_for_events(events: List[Dict[str, Any]], pre_days: int = 180, post_days: int = 365, force: bool = False):
    events_mod = []
    for ev in tqdm(events, desc="fetch_prices"):
        ticker = ev.get("ticker")
        if not ticker:
            continue
        filing_date = dt.datetime.strptime(ev['filingDate'], "%Y-%m-%d").date()
        start = (filing_date - dt.timedelta(days=pre_days)).isoformat()
        end = (filing_date + dt.timedelta(days=post_days)).isoformat()
        df = tiingo_price_fetch(ticker, start, end, force=force)
        if df is not None:
            ev_copy = dict(ev)
            ev_copy['price_slice'] = str(PRICES_DIR / f"{ticker}__{start}__{end}.csv")
            events_mod.append(ev_copy)
        else:
            logger.warning("Prices missing for %s around %s", ticker, ev['filingDate'])
    json.dump(events_mod, open(EVENTS_FILE, "w"), indent=2, default=str)
    return events_mod

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

    trough_price = float(post[price_col].min())
    trough_idx = post[price_col].idxmin()
    feat['trough_price'] = trough_price
    feat['trough_depth_pct'] = float((trough_price - price_at_t0) / price_at_t0)
    feat['time_to_trough_days'] = int((trough_idx - t0).days)

    rec_mask = post[post[price_col] >= price_at_t0]
    if not rec_mask.empty:
        rec_idx = rec_mask.index[0]
        feat['recovered_within_post_days'] = 1
        feat['days_from_t0_to_recovery'] = int((rec_idx - t0).days)
        feat['days_from_trough_to_recovery'] = int((rec_idx - trough_idx).days)
    else:
        feat['recovered_within_post_days'] = 0
        feat['days_from_t0_to_recovery'] = None
        feat['days_from_trough_to_recovery'] = None

    post_after_trough = post.loc[trough_idx:]
    if len(post_after_trough) >= 3:
        x = np.arange(len(post_after_trough))
        y = post_after_trough[price_col].values
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        feat['post_trough_slope'] = float(m)
    else:
        feat['post_trough_slope'] = 0.0

    return feat

def featurize_events(pre_window_days: int = 180, post_window_days: int = 365, force: bool = False):
    if not EVENTS_FILE.exists():
        logger.error("No events found. Run discover first.")
        return None
    events = json.loads(EVENTS_FILE.read_text())
    features = []
    for ev in tqdm(events, desc="featurize"):
        out = compute_event_features(ev, pre_window_days, post_window_days)
        if out:
            features.append(out)
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
def train_models(force: bool = False):
    if not Path(FEATURES_FILE).exists():
        logger.error("No features found. Run featurize first.")
        return None
    df = pd.read_pickle(FEATURES_FILE)
    df = df.dropna(subset=['recovered_within_post_days'])
    y = df['recovered_within_post_days'].astype(int)
    feature_cols = ['pre_mean_ret','pre_vol','pre_momentum_30d','pre_ad_mean_volume','pre_max_drawdown','trough_depth_pct','time_to_trough_days']
    X = df[feature_cols].fillna(0).astype(float)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1))
    ])

    tscv = TimeSeriesSplit(n_splits=3)
    try:
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="roc_auc", n_jobs=1)
        logger.info("Cross-validated ROC-AUC: %.4f ± %.4f", scores.mean(), scores.std())
    except Exception as e:
        logger.warning("CV failed: %s", e)

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODELS_DIR / "classifier.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    logger.info("Saved classifier to %s", MODELS_DIR / "classifier.joblib")

    if LIFELINES_AVAILABLE:
        surv_df = df.copy()
        surv_df['duration'] = surv_df['days_from_t0_to_recovery'].fillna(surv_df['post_days']).astype(float)
        surv_df['event'] = surv_df['recovered_within_post_days'].astype(int)
        cols_for_surv = ['pre_mean_ret','pre_vol','pre_momentum_30d','pre_ad_mean_volume','pre_max_drawdown','trough_depth_pct']
        surv_in = surv_df[cols_for_surv + ['duration','event']].fillna(0)
        cph = CoxPHFitter()
        try:
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
    for ev in tqdm(events, desc="scoring"):
        feat = compute_event_features(ev)
        if not feat:
            continue
        Xrow = pd.DataFrame([feat])[feature_cols].fillna(0)
        prob = clf.predict_proba(Xrow)[0, 1]
        rows.append({**feat, "prob_recovery": float(prob)})
    return pd.DataFrame(rows).sort_values("prob_recovery", ascending=False) if rows else pd.DataFrame()

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

    f = sub.add_parser("fetch_prices", help="Fetch Tiingo prices for discovered events")
    f.add_argument("--pre-days", type=int, default=180)
    f.add_argument("--post-days", type=int, default=365)
    f.add_argument("--force", action="store_true")
    f.add_argument("--log", type=str, default="INFO")

    feat = sub.add_parser("featurize", help="Compute features for events")
    feat.add_argument("--pre-days", type=int, default=180)
    feat.add_argument("--post-days", type=int, default=365)
    feat.add_argument("--log", type=str, default="INFO")

    train = sub.add_parser("train", help="Train ML models")
    train.add_argument("--force", action="store_true")
    train.add_argument("--log", type=str, default="INFO")

    score = sub.add_parser("score_universe", help="Score the cached events or all tickers")
    score.add_argument("--mode", choices=["cached_events", "predict_all"], default="cached_events")
    score.add_argument("--log", type=str, default="INFO")

    sp = sub.add_parser("show_proxies", help="Show loaded proxies and exit")
    sp.add_argument("--log", type=str, default="INFO")

    # New: quarterly index discovery (robust, no efts usage)
    di = sub.add_parser("discover_index", help="Discover via quarterly full-index master.idx")
    di.add_argument("--years", type=int, default=5)
    di.add_argument("--throttle", type=float, default=0.1)
    di.add_argument("--strict", action="store_true")
    di.add_argument("--force", action="store_true")
    di.add_argument("--log", type=str, default="INFO")
    di.add_argument("--price-only", action="store_true", help="Keep only price/market-value related Item 3.01 events")

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
        )
    elif args.cmd == "discover_index":
        discover_via_index(
            years=args.years,
            throttle=args.throttle,
            force=args.force,
            strict=args.strict,
            price_only=getattr(args, "price_only", False),
        )
    elif args.cmd == "fetch_prices":
        if not EVENTS_FILE.exists():
            logger.error("No events.json found; run discover first.")
            sys.exit(1)
        events = json.loads(EVENTS_FILE.read_text())
        fetch_prices_for_events(events, pre_days=args.pre_days, post_days=args.post_days, force=args.force)
    elif args.cmd == "featurize":
        featurize_events(pre_window_days=args.pre_days, post_window_days=args.post_days)
    elif args.cmd == "train":
        train_models(force=args.force)
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
    elif args.cmd == "show_proxies":
        for i, pxy in enumerate(PROXIES, 1):
            logger.info("[%02d] %s", i, pxy.get("_display", "direct"))
    else:
        print("No command provided. Use --help.")

if __name__ == "__main__":
    main()
