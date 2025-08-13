# scanner.py
# Free, automated alerts using Yahoo Finance + Telegram, scheduled by GitHub Actions.
# Tracks: SPY, AAPL, CVX, AMZN, QQQ, GLD, SLV, PLTR, USO, NFLX, TNA, XOM, NVDA, BAC, TSLA, META
# Signals: RSI(14), EMA(20), simple Hammer/Hanging-Man, options strike suggestions (delta-based if IV available)
# Sends: Buy/Sell suggestions, Schwab-fill helper, ITM alerts, progress updates
# State: persists alerts_state.json in repo to avoid duplicate pings

import os, json, math, time, sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pathlib import Path
from math import log, sqrt, exp
from statistics import mean

# ----------------------------
# Settings
# ----------------------------
WATCHLIST = ["SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR","USO","NFLX","TNA","XOM","NVDA","BAC","TSLA","META"]
EMA_WINDOW = 20
RSI_WINDOW = 14
TARGET_DELTA = 0.35   # ~30–40 delta options
MIN_DTE = 14          # pick expirations at least 14 days out when possible
MAX_DTE = 60          # cap at ~2 months
RISK_FREE = 0.04      # 4% annualized (rough); used for delta approx
STATE_FILE = "alerts_state.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

UTC_NOW = datetime.now(timezone.utc)

# ----------------------------
# Helpers
# ----------------------------
def send_telegram(msg: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Telegram tokens missing; skipping send.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("Telegram error:", e, file=sys.stderr)

def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def rsi(series: pd.Series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(window).mean()
    loss = pd.Series(loss, index=series.index).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def detect_hammer_like(o,h,l,c):
    body = abs(c-o)
    rng = max(h-l, 1e-9)
    lower = min(o,c) - l
    upper = h - max(o,c)
    # hammer-ish shape
    return (lower >= 2.0*max(body,1e-9)) and (upper <= 0.5*max(body,1e-9)) and (body/rng <= 0.35)

def bs_call_delta(S, K, T, r, sigma):
    # Basic Black-Scholes call delta (no dividends)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: 
        return None
    try:
        d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        # standard normal CDF
        return 0.5*(1+math.erf(d1/math.sqrt(2)))
    except Exception:
        return None

def choose_expiration(expirations, min_dte=MIN_DTE, max_dte=MAX_DTE):
    if not expirations:
        return None
    today = UTC_NOW.date()
    candidates = []
    for exp in expirations:
        try:
            ed = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (ed - today).days
            if min_dte <= dte <= max_dte:
                candidates.append((dte, exp))
        except Exception:
            continue
    if candidates:
        candidates.sort()
        return candidates[0][1]  # nearest >= min_dte
    # fallback: nearest available
    return expirations[0]

def nearest_delta_option(chain_df, spot, target_delta=TARGET_DELTA, is_call=True):
    if chain_df is None or chain_df.empty:
        return None
    # yfinance provides impliedVolatility (as decimal), bid/ask/lastPrice, strike, contractSymbol
    df = chain_df.copy()
    if "impliedVolatility" not in df.columns:
        return None
    df["iv"] = df["impliedVolatility"].astype(float)
    df["strike"] = df["strike"].astype(float)
    # approximate T in years from expiry embedded in contractSymbol if present (fallback to ~30 days)
    # yfinance doesn't hand us T directly here; we compute outside and pass in ideally.
    # For delta selection we'll approximate with a constant T=30d if unknown.
    T_years = 30/365
    # compute delta approx
    deltas = []
    for _, row in df.iterrows():
        sigma = row["iv"]
        K = row["strike"]
        d = bs_call_delta(spot, K, T_years, RISK_FREE, sigma)
        if d is None: 
            deltas.append(np.nan)
            continue
        deltas.append(d)
    df["delta_approx"] = deltas
    # For puts, transform target (put delta ≈ call delta -1)
    if not is_call:
        # approximate put delta with call delta - 1
        df["delta_approx"] = df["delta_approx"] - 1.0
        target = -TARGET_DELTA
    else:
        target = TARGET_DELTA

    # choose row with delta closest to target
    df["delta_diff"] = (df["delta_approx"] - target).abs()
    df = df.sort_values("delta_diff")
    if df.empty:
        return None
    row = df.iloc[0]
    # mid price suggestion
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    mid = None
    if not np.isnan(bid) and not np.isnan(ask) and ask >= bid and bid > 0:
        mid = round((bid + ask)/2, 2)
    return {
        "contract": row.get("contractSymbol", ""),
        "strike": float(row["strike"]),
        "iv": float(row["iv"]) if not math.isnan(float(row["iv"])) else None,
        "bid": float(bid) if not math.isnan(bid) else None,
        "ask": float(ask) if not math.isnan(ask) else None,
        "mid": mid,
        "delta_approx": float(row["delta_approx"]) if not math.isnan(row["delta_approx"]) else None
    }

def schwab_fields(tkr, contract_dict, is_call, qty=1, limit=None):
    # Schwab helper fields to fill
    cp = "CALL" if is_call else "PUT"
    return (
        f"Schwab fields:\n"
        f"• Action: BUY TO OPEN\n"
        f"• Quantity: {qty}\n"
        f"• Symbol: {tkr}\n"
        f"• Option Type: {cp}\n"
        f"• Strike: {contract_dict.get('strike')}\n"
        f"• Expiration: (from contract symbol)\n"
        f"• Order type: LIMIT\n"
        f"• Limit price: {limit if limit is not None else contract_dict.get('mid')}\n"
        f"• Time-in-force: DAY\n"
    )

def analyze_ticker(tkr: str):
    tk = yf.Ticker(tkr)
    hist = tk.history(period="6mo", interval="1d")
    if hist.empty:
        return {"ticker": tkr, "error": "no price data"}
    hist = hist.dropna()
    hist["EMA20"] = hist["Close"].ewm(span=EMA_WINDOW, adjust=False).mean()
    hist["RSI14"] = rsi(hist["Close"], RSI_WINDOW)

    last = hist.iloc[-1]
    close, ema, rsi_v = float(last["Close"]), float(last["EMA20"]), float(last["RSI14"])
    # simple pattern (last candle only)
    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    hammerish = detect_hammer_like(o,h,l,c)

    # Context + signal
    bullish = (rsi_v < 30 and close > ema) or (hammerish and close > ema)
    bearish = (rsi_v > 70 and close < ema) or (hammerish and close < ema)  # hanging-like context

    signal = "HOLD"
    if bullish and not bearish:
        signal = "BUY_CALL"
    elif bearish and not bullish:
        signal = "BUY_PUT"

    # options suggestion
    expirations = tk.options
    if not expirations:
        return {"ticker": tkr, "close": close, "ema20": ema, "rsi14": rsi_v, "signal": signal, "error":"no options"}
    exp = choose_expiration(expirations, MIN_DTE, MAX_DTE)
    if not exp:
        return {"ticker": tkr, "close": close, "ema20": ema, "rsi14": rsi_v, "signal": signal, "error":"no exp chosen"}

    chain = tk.option_chain(exp)
    calls = chain.calls if hasattr(chain, "calls") else None
    puts  = chain.puts  if hasattr(chain, "puts") else None

    if signal == "BUY_CALL":
        sel = nearest_delta_option(calls, close, TARGET_DELTA, is_call=True)
    elif signal == "BUY_PUT":
        sel = nearest_delta_option(puts, close, TARGET_DELTA, is_call=False)
    else:
        sel = None

    # ITM detection for suggested contract (or for all close-to-ATM if none)
    itm = None
    if sel:
        if signal == "BUY_CALL":
            itm = close >= sel["strike"]
        elif signal == "BUY_PUT":
            itm = close <= sel["strike"]

    # basic exit guidance (you can tune later)
    # - Take profit at +50% from entry
    # - Stop at -30% from entry
    # - Time-based: consider exiting 7-10 days before expiration
    guidance = (
        "Exit guide: take profit near +50%; cut loss near -30%; recheck IV & trend; "
        "avoid holding inside last 7–10 days to expiration unless strongly ITM."
    )

    return {
        "ticker": tkr,
        "close": close,
        "ema20": ema,
        "rsi14": rsi_v,
        "signal": signal,
        "expiration": exp,
        "selection": sel,    # dict with strike, mid, bid/ask, contractSymbol, delta_approx
        "itm": itm,
        "hammer_like": hammerish,
        "guidance": guidance
    }

def make_message(res):
    t = res["ticker"]
    s = res["signal"]
    price = res["close"]
    ema = res["ema20"]; r = res["rsi14"]
    exp = res.get("expiration")
    sel = res.get("selection")
    base = f"{t} @ {price:.2f} | RSI {r:.1f} | EMA20 {ema:.2f}"
    if s == "BUY_CALL" and sel:
        cp = "CALL"
        msg = (f"📈 {t} BUY {cp} signal\n"
               f"{base}\n"
               f"Exp: {exp} | Strike ~ {sel['strike']:.2f} | Δ≈{(sel['delta_approx'] or 0):.2f}\n"
               f"Mid≈ {sel['mid']} (bid {sel['bid']}, ask {sel['ask']})\n"
               f"OCC: {sel['contract']}\n"
               f"{res['guidance']}\n"
               f"ITM now? {'YES ✅' if res['itm'] else 'No'}\n\n"
               f"{schwab_fields(t, sel, is_call=True, limit=sel['mid'])}")
    elif s == "BUY_PUT" and sel:
        cp = "PUT"
        msg = (f"📉 {t} BUY {cp} signal\n"
               f"{base}\n"
               f"Exp: {exp} | Strike ~ {sel['strike']:.2f} | Δ≈{(sel['delta_approx'] or 0):.2f}\n"
               f"Mid≈ {sel['mid']} (bid {sel['bid']}, ask {sel['ask']})\n"
               f"OCC: {sel['contract']}\n"
               f"{res['guidance']}\n"
               f"ITM now? {'YES ✅' if res['itm'] else 'No'}\n\n"
               f"{schwab_fields(t, sel, is_call=False, limit=sel['mid'])}")
    else:
        msg = f"⏸ {t} HOLD | {base}"
    return msg

def main():
    state = load_state()
    changed = False
    summaries = []
    for t in WATCHLIST:
        try:
            res = analyze_ticker(t)
        except Exception as e:
            print(f"{t} error: {e}", file=sys.stderr)
            continue

        key = f"{t}"
        prev = state.get(key, {})
        # Compose compact status for dedupe
        sel = res.get("selection") or {}
        status = {
            "signal": res.get("signal"),
            "exp": res.get("expiration"),
            "strike": sel.get("strike"),
            "itm": bool(res.get("itm"))
        }

        # Send alerts only on change OR when ITM flips to True
        should_alert = False
        if prev != status:
            should_alert = True
        else:
            # If already same status but just turned ITM (edge case)
            if (not prev.get("itm", False)) and status.get("itm", False):
                should_alert = True

        if should_alert:
            msg = make_message(res)
            send_telegram(msg)
            state[key] = status
            changed = True

        # Keep a short summary for logs
        summaries.append(f"{t}: {status['signal']} exp={status['exp']} strike={status['strike']} ITM={status['itm']}")

        # print to console too
        print(summaries[-1])

    if changed:
        save_state(state)
        print("State updated.")
    else:
        print("No changes; state unchanged.")

if __name__ == "__main__":
    main()
