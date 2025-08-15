import os, json, time
from datetime import datetime, timedelta, timezone
import requests
import yfinance as yf
import pandas as pd

# ========= USER SETTINGS =========
TICKERS = [
    "SPY","AAPL","CVX","AMZN","QQQ","GLD","SLV","PLTR","USO",
    "NFLX","TNA","XOM","NVDA","BAC","TSLA","META"
]

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT = 9
EMA_LONG  = 21

TAKE_PROFIT_PCT = 0.30   # +30% on option mid
STOP_LOSS_PCT   = -0.20  # -20% on option mid

STATE_FILE = "state.json"  # keeps last alerts + open positions, prevents spam
# =================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def send(msg: str):
    if not (BOT_TOKEN and CHAT_ID):
        print("Missing TELEGRAM env vars.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram send failed:", e)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"last_alerts": {}, "positions": {}}
    return {"last_alerts": {}, "positions": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_latest_indicators(ticker: str):
    # Use 30m bars over 2 months (free via yfinance)
    df = yf.download(ticker, period="2mo", interval="30m", progress=False)
    if df.empty:
        return None
    close = df["Close"].dropna()
    ema_s = close.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_l = close.ewm(span=EMA_LONG, adjust=False).mean()
    r = rsi(close, RSI_PERIOD)
    return {
        "price": float(close.iloc[-1]),
        "ema_s": float(ema_s.iloc[-1]),
        "ema_l": float(ema_l.iloc[-1]),
        "rsi":   float(r.iloc[-1]),
    }

def choose_expiry(tkr_obj: yf.Ticker, min_days=7):
    today = datetime.now(timezone.utc).date()
    expiries = tkr_obj.options or []
    if not expiries:
        return None
    # pick first expiry at least min_days out; else the last available
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            if (d - today).days >= min_days:
                return e
        except Exception:
            continue
    return expiries[-1]  # fallback

def pick_contract(ticker: str, opt_type: str, underlying_price: float):
    t = yf.Ticker(ticker)
    expiry = choose_expiry(t, min_days=7)
    if not expiry:
        return None

    chain = t.option_chain(expiry)
    table = chain.calls if opt_type == "CALL" else chain.puts
    if table.empty:
        return None

    # pick ATM: strike closest to underlying
    table = table.copy()
    table["dist"] = (table["strike"] - underlying_price).abs()
    row = table.sort_values("dist").iloc[0]

    # Safe mid calc
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    last = float(row.get("lastPrice", 0) or 0)
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    else:
        mid = last if last > 0 else max(bid, ask)

    return {
        "contract": row.get("contractSymbol", ""),
        "expiry":   expiry,
        "strike":   float(row["strike"]),
        "type":     opt_type,
        "bid":      bid,
        "ask":      ask,
        "mid":      float(mid) if mid else None,
        "itm":      bool(row.get("inTheMoney", False))
    }

def signal_from_indicators(ind):
    # BUY CALLS: oversold + bullish EMA
    if ind["rsi"] < RSI_OVERSOLD and ind["ema_s"] > ind["ema_l"]:
        return "BUY_CALLS"
    # BUY PUTS: overbought + bearish EMA
    if ind["rsi"] > RSI_OVERBOUGHT and ind["ema_s"] < ind["ema_l"]:
        return "BUY_PUTS"
    return None

def open_or_update_positions(state, ticker, option):
    """
    Track one open suggestion per ticker.
    Store entry mid so we can monitor P/L and ITM.
    """
    pos = state["positions"].get(ticker)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not pos:
        # open new tracked position
        state["positions"][ticker] = {
            "contract":      option["contract"],
            "type":          option["type"],     # CALL or PUT
            "strike":        option["strike"],
            "expiry":        option["expiry"],
            "entry_mid":     option["mid"],
            "underlying_at": None,
            "opened_at":     now,
            "itm_notified":  False,
            "tp_notified":   False,
            "sl_notified":   False
        }
        return True
    else:
        # same contract? If expiry or strike changed, replace (new idea)
        changed = (pos["contract"] != option["contract"])
        if changed:
            state["positions"][ticker] = {
                "contract":      option["contract"],
                "type":          option["type"],
                "strike":        option["strike"],
                "expiry":        option["expiry"],
                "entry_mid":     option["mid"],
                "underlying_at": None,
                "opened_at":     now,
                "itm_notified":  False,
                "tp_notified":   False,
                "sl_notified":   False
            }
        return changed

def current_option_mid(ticker, contract_symbol, expiry, opt_type, strike):
    # re-pull chain for current mid
    t = yf.Ticker(ticker)
    try:
        chain = t.option_chain(expiry)
        tab = chain.calls if opt_type == "CALL" else chain.puts
        row = tab.loc[tab["contractSymbol"] == contract_symbol]
        if row.empty:
            # fallback by strike/type if OCC symbol rotated
            row = tab.loc[abs(tab["strike"] - strike) < 1e-6]
            if row.empty:
                return None
        bid = float(row.iloc[0].get("bid", 0) or 0)
        ask = float(row.iloc[0].get("ask", 0) or 0)
        last= float(row.iloc[0].get("lastPrice", 0) or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return last if last > 0 else max(bid, ask)
    except Exception:
        return None

def check_itm_and_risk(state, ticker, underlying_price):
    """
    ITM check + TP/SL guidance. One-time notifications to avoid spam.
    """
    pos = state["positions"].get(ticker)
    if not pos:
        return

    # ITM
    if pos["type"] == "CALL":
        is_itm = underlying_price >= pos["strike"]
    else:
        is_itm = underlying_price <= pos["strike"]

    if is_itm and not pos["itm_notified"]:
        send(f"✅ ITM: {ticker} {pos['type']} {pos['expiry']} {pos['strike']}\nSpot: {underlying_price:.2f}")
        pos["itm_notified"] = True

    # TP/SL on option mid
    cur_mid = current_option_mid(ticker, pos["contract"], pos["expiry"], pos["type"], pos["strike"])
    if cur_mid and pos["entry_mid"] and pos["entry_mid"] > 0:
        pnl = (cur_mid / pos["entry_mid"]) - 1.0
        if pnl >= TAKE_PROFIT_PCT and not pos["tp_notified"]:
            send(f"📈 Take-Profit hit on {ticker} {pos['type']} {pos['expiry']} {pos['strike']}\n"
                 f"Entry mid: {pos['entry_mid']:.2f} → Now: {cur_mid:.2f} (+{pnl*100:.1f}%).\n"
                 f"Suggestion: Consider SELL to realize gains.")
            pos["tp_notified"] = True
        if pnl <= STOP_LOSS_PCT and not pos["sl_notified"]:
            send(f"⚠️ Stop-Loss hit on {ticker} {pos['type']} {pos['expiry']} {pos['strike']}\n"
                 f"Entry mid: {pos['entry_mid']:.2f} → Now: {cur_mid:.2f} ({pnl*100:.1f}%).\n"
                 f"Suggestion: Consider SELL to limit loss.")
            pos["sl_notified"] = True

def format_order_instructions(ticker, opt):
    """
    What to fill on Schwab (or any broker):
    - Action: BUY TO OPEN
    - Quantity: 1 (example)
    - Symbol: {ticker} options → pick expiry + strike + Call/Put
    - Order Type: LIMIT
    - Limit Price: suggested mid
    - Time in Force: DAY
    """
    mid_txt = f"{opt['mid']:.2f}" if opt["mid"] else "set reasonable limit"
    return (
        f"➡️ Schwab fields for {ticker}:\n"
        f"• Action: BUY TO OPEN\n"
        f"• Option Type: {opt['type']}\n"
        f"• Expiration: {opt['expiry']}\n"
        f"• Strike: {opt['strike']}\n"
        f"• Order Type: LIMIT\n"
        f"• Limit Price: ~{mid_txt}\n"
        f"• Time in Force: DAY\n"
        f"(You can also search contract symbol: {opt['contract']})"
    )

def main():
    state = load_state()
    last_alerts = state.get("last_alerts", {})
    changes = []

    for tk in TICKERS:
        ind = get_latest_indicators(tk)
        if not ind:
            continue

        sig = signal_from_indicators(ind)
        price = ind["price"]

        # Always check ITM/TP/SL for any tracked position
        check_itm_and_risk(state, tk, price)

        # No new directional signal → continue
        if not sig:
            continue

        # Map signal to option type
        opt_type = "CALL" if sig == "BUY_CALLS" else "PUT"

        # Build a concise signature so we only alert on CHANGE
        signature = f"{sig}|{tk}"

        if last_alerts.get(tk) == signature:
            # same idea already sent → skip (no spam)
            continue

        # Pick contract (ATM, >=7 DTE)
        opt = pick_contract(tk, opt_type, price)
        if not opt:
            continue

        # Prepare message
        msg = (
            f"📣 {tk}: { 'BUY CALLS' if opt_type=='CALL' else 'BUY PUTS' }\n"
            f"• Price: {price:.2f}\n"
            f"• RSI/EMA confirm (RSI {ind['rsi']:.1f}, EMA{EMA_SHORT}>{EMA_LONG} bullish = CALLS; "
            f"RSI {RSI_OVERBOUGHT}/{RSI_OVERSOLD} & EMA cross bearish = PUTS)\n\n"
            f"Suggested contract:\n"
            f"• {opt_type} {opt['expiry']} ${opt['strike']:.2f}\n"
            f"• Mid≈ {opt['mid']:.2f}  (bid {opt['bid']:.2f} / ask {opt['ask']:.2f})\n"
            f"• Contract: {opt['contract']}\n\n"
            f"{format_order_instructions(tk, opt)}\n\n"
            f"Risk guide (tracked): TP +30%, SL −20%, ITM ping active."
        )
        send(msg)

        # Save last alert signature and track the suggested position
        last_alerts[tk] = signature
        opened = open_or_update_positions(state, tk, opt)
        if opened:
            send(f"🔎 Tracking {tk} {opt['type']} {opt['expiry']} {opt['strike']} at entry mid≈ {opt['mid']:.2f}.")

    # Persist
    state["last_alerts"] = last_alerts
    save_state(state)

if __name__ == "__main__":
    main()
