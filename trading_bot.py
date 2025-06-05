"""
Updated Bybit unified‑trading bot – single‑file edition
======================================================
• Risk‑based sizing (percent of **total equity**)  
• 5 % hard stop, 10 % / 10 % trailing stop  
• Two‑tier grid TP unchanged but bug‑fixed  
• SQLite trade ledger via SQLAlchemy  
• Full Flask + Telegram webhook interface  

Test on Bybit **testnet** first – set `TESTNET=True` in the env file.
"""

from __future__ import annotations

import os
import math
import logging
import time
from functools import lru_cache
from threading import Lock
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import telebot
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from pybit.unified_trading import HTTP  # Official Bybit v5 library

# ---------------------------------------------------------------------------
#                   0)  Configuration & Bootstrapping
# ---------------------------------------------------------------------------

load_dotenv()

RISK_PCT = float(os.getenv("RISK_PCT", 1.0))          # % of equity per new trade
LEVERAGE = os.getenv("LEVERAGE", "25")                # x‑leverage string (Bybit API wants str)
STOP_LOSS_PCT = 0.05                                    # 5 % hard SL
TRAILING_TRIGGER = 0.10                                 # 10 % profit before trailing activates
TRAILING_OFFSET = 10                                    # Bybit offset, integer %
REQUEST_INTERVAL = 0.1                                  # default throttle (s)
DB_URL = os.getenv("DB_URL", "sqlite:///trades.db")
TESTNET = os.getenv("TESTNET", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

lock = Lock()

# ---------------------------------------------------------------------------
#                   1)  Exchange Session & Telegram
# ---------------------------------------------------------------------------

session = HTTP(
    testnet=TESTNET,
    api_key=os.getenv("BYBIT_API_KEY", "YOUR_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET", "YOUR_API_SECRET"),
)

bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN"))
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

app = Flask(__name__)

# ---------------------------------------------------------------------------
#                   2)  Database – SQLAlchemy ORM
# ---------------------------------------------------------------------------

engine = create_engine(DB_URL, echo=False, future=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    side = Column(String)  # LONG or SHORT
    qty = Column(Float)
    entry_px = Column(Float)
    exit_px = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    pnl = Column(Float, nullable=True)


Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
#                   3)  Utility wrappers – rate limit & retries
# ---------------------------------------------------------------------------

def _rate_limited_request(func, *args, **kwargs):
    """Headers‑aware throttling – single instance shared across threads."""
    with lock:
        response = func(*args, **kwargs)
        headers = response.get("headers", {})
        remaining_calls = int(headers.get("X-Bapi-Limit-Status", 10))
        reset_ts = int(headers.get("X-Bapi-Limit-Reset-Timestamp", 0))

        if remaining_calls < 5:
            sleep_t = max(2, reset_ts - int(time.time()))
            logger.warning("Approaching rate limit ({}). Sleeping {} s".format(remaining_calls, sleep_t))
            time.sleep(sleep_t)
        elif remaining_calls < 20:
            time.sleep(1)
        else:
            time.sleep(REQUEST_INTERVAL)
        return response


def safe_api_call(func, *args, retries: int = 5, **kwargs):
    for attempt in range(retries):
        try:
            return _rate_limited_request(func, *args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower():
                backoff = 2 ** attempt  # exponential
                logger.warning(f"Rate‑limit on attempt {attempt + 1}; retrying in {backoff}s")
                time.sleep(backoff)
            else:
                raise
    logger.error(f"{func.__name__} failed after {retries} retries.")
    return None


# ---------------------------------------------------------------------------
#                   4)  Instrument meta & helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def fetch_qty_limits(symbol: str):
    resp = safe_api_call(session.get_instruments_info, category="linear", symbol=symbol)
    if resp["retCode"] != 0:
        logger.error(f"get_instruments_info error: {resp}")
        return {"min": 0.01, "max": 999999, "step": 0.01, "pricePrecision": 2, "tickSize": 0.01}

    data = resp["result"]["list"][0]
    lot = data["lotSizeFilter"]
    price_f = data["priceFilter"]
    return {
        "min": float(lot["minOrderQty"]),
        "max": float(lot["maxOrderQty"]),
        "step": float(lot["qtyStep"]),
        "pricePrecision": int(-math.log10(float(price_f["tickSize"]))),
        "tickSize": float(price_f["tickSize"]),
    }


def clamp_and_step(qty: float, lims: dict) -> float:
    clamped = max(lims["min"], min(qty, lims["max"]))
    step_cnt = math.floor(clamped / lims["step"])
    return round(step_cnt * lims["step"], 8)


def round_price(price: float, precision: int) -> float:
    multiplier = 10 ** precision
    return math.floor(price * multiplier) / multiplier


# ---------------------------------------------------------------------------
#                   5)  Price & position data
# ---------------------------------------------------------------------------

def get_current_price(symbol: str) -> float | None:
    resp = safe_api_call(session.get_tickers, category="linear", symbol=symbol)
    if resp["retCode"] != 0 or not resp["result"]["list"]:
        return None
    return float(resp["result"]["list"][0]["lastPrice"])


def get_account_equity_usd() -> float:
    resp = safe_api_call(session.get_wallet_balance, accountType="UNIFIED")
    if resp["retCode"] != 0:
        logger.error(f"wallet_balance error: {resp}")
        return 0.0
    return float(resp["result"]["list"][0]["totalEquity"])


def calculate_position_quantity(symbol: str) -> float:
    equity = get_account_equity_usd()
    price = get_current_price(symbol)
    if equity == 0 or price is None:
        return 0.0
    risk_dollars = equity * RISK_PCT / 100
    raw_qty = risk_dollars / (price * STOP_LOSS_PCT)
    lims = fetch_qty_limits(symbol)
    final = clamp_and_step(raw_qty, lims)
    logger.debug(f"[calc_qty] {symbol} raw={raw_qty}, final={final}, lims={lims}")
    return final


def get_position_entry_price(symbol: str, side: str) -> float | None:
    bybit_side = "Buy" if side == "LONG" else "Sell"
    resp = safe_api_call(session.get_positions, category="linear", symbol=symbol)
    if resp["retCode"] != 0:
        return None
    for p in resp["result"]["list"]:
        if p["symbol"] == symbol and p["side"] == bybit_side and float(p["size"]) > 0:
            return float(p["avgPrice"])
    return None


# ---------------------------------------------------------------------------
#                   6)  Trade‑logging helpers
# ---------------------------------------------------------------------------

def log_trade(symbol: str, side: str, qty: float, entry_px: float):
    with SessionLocal() as db:
        db.add(Trade(symbol=symbol, side=side, qty=qty, entry_px=entry_px))
        db.commit()


def mark_trade_closed(symbol: str, side: str, exit_px: float):
    with SessionLocal() as db:
        tr = (
            db.query(Trade)
            .filter_by(symbol=symbol, side=side, exit_px=None)
            .order_by(Trade.opened_at.desc())
            .first()
        )
        if tr:
            tr.exit_px = exit_px
            tr.closed_at = datetime.utcnow()
            tr.pnl = (exit_px - tr.entry_px) * tr.qty * (1 if side == "LONG" else -1)
            db.commit()


# ---------------------------------------------------------------------------
#                   7)  Risk controls – SL & trailing
# ---------------------------------------------------------------------------

def _set_tp_sl(symbol: str, pos_idx: int, **kwargs):
    safe_api_call(session.set_trading_stop,
                  category="linear",
                  symbol=symbol,
                  positionIdx=pos_idx,
                  tpslMode="Full",
                  **kwargs)


def set_hard_stop_loss(symbol: str, side: str, entry_px: float):
    stop_px = entry_px * (1 - STOP_LOSS_PCT) if side == "LONG" else entry_px * (1 + STOP_LOSS_PCT)
    precision = fetch_qty_limits(symbol)["pricePrecision"]
    _set_tp_sl(symbol, 1 if side == "LONG" else 2, stopLoss=str(round_price(stop_px, precision)))


def set_trailing_stop(symbol: str, side: str, entry_px: float):
    trigger_px = entry_px * (1 + TRAILING_TRIGGER) if side == "LONG" else entry_px * (1 - TRAILING_TRIGGER)
    precision = fetch_qty_limits(symbol)["pricePrecision"]
    _set_tp_sl(
        symbol,
        1 if side == "LONG" else 2,
        trailingStop=str(TRAILING_OFFSET),
        triggerPrice=str(round_price(trigger_px, precision)),
    )


# ---------------------------------------------------------------------------
#                   8)  Grid Take‑Profit function (80 % ladder + 20 % bank)
# ---------------------------------------------------------------------------

def set_two_tier_take_profits(symbol: str, side: str, total_qty: float, entry_px: float):
    steps = 40
    partial_qty = total_qty * 0.8
    remaining_qty = total_qty * 0.2
    lims = fetch_qty_limits(symbol)
    precision = lims["pricePrecision"]
    tick = lims["tickSize"]
    pos_idx = 1 if side == "LONG" else 2

    step_qty = clamp_and_step(partial_qty / steps, lims)

    if side == "LONG":
        inc = (1.40 - 1.01) / (steps - 1)
        price_fn = lambda i: entry_px * (1.01 + i * inc)
        tp_side = "Sell"
    else:
        inc = (0.99 - 0.60) / (steps - 1)
        price_fn = lambda i: entry_px * (0.99 - i * inc)
        tp_side = "Buy"

    orders = []
    for i in range(steps):
        px = round(price_fn(i) / tick) * tick
        orders.append({
            "symbol": symbol,
            "side": tp_side,
            "orderType": "Limit",
            "qty": str(step_qty),
            "price": str(round(px, precision)),
            "timeInForce": "GTC",
            "reduceOnly": True,
            "positionIdx": pos_idx,
        })

    final_px = entry_px * (1.50 if side == "LONG" else 0.50)
    final_px = round(final_px / tick) * tick
    final_qty = clamp_and_step(remaining_qty, lims)
    orders.append({
        "symbol": symbol,
        "side": tp_side,
        "orderType": "Limit",
        "qty": str(final_qty),
        "price": str(round(final_px, precision)),
        "timeInForce": "GTC",
        "reduceOnly": True,
        "positionIdx": pos_idx,
    })

    _place_batch_orders(orders)


def _place_batch_orders(orders: list[dict], retries: int = 3):
    chunk = 10
    for i in range(0, len(orders), chunk):
        sub = orders[i:i + chunk]
        for attempt in range(retries):
            resp = safe_api_call(session.place_batch_order, category="linear", request=sub)
            if resp and resp.get("retCode") == 0:
                break
            logger.error(f"Batch error: {resp}. Retry {attempt + 1}")
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
#                   9)  Position management (open / close)
# ---------------------------------------------------------------------------

def ensure_hedge_mode(symbol: str | None = None):
    params = {"category": "linear", "mode": 3}
    if symbol:
        params["symbol"] = symbol
    else:
        params["settleCoin"] = "USDT"
    safe_api_call(session.switch_position_mode, params)


def set_leverage(symbol: str):
    resp = safe_api_call(session.get_positions, category="linear", symbol=symbol)
    if resp["retCode"] == 0 and resp["result"]["list"]:
        if resp["result"]["list"][0]["leverage"] == LEVERAGE:
            return
    safe_api_call(session.set_leverage, category="linear", symbol=symbol,
                  buyLeverage=LEVERAGE, sellLeverage=LEVERAGE)


def open_position(symbol: str, buy_sell: str, side: str):
    qty = calculate_position_quantity(symbol)
    if qty <= 0:
        logger.error("Qty calc produced 0 – aborting open.")
        return

    ensure_hedge_mode(symbol)
    set_leverage(symbol)

    pos_idx = 1 if side == "LONG" else 2
    order = safe_api_call(session.place_order, category="linear", symbol=symbol,
                          side="Buy" if buy_sell == "BUY" else "Sell",
                          orderType="Market", qty=str(qty), timeInForce="GTC", positionIdx=pos_idx)
    if order["retCode"] != 0:
        logger.error(f"Open {side} error: {order}")
        return

    entry_px = get_position_entry_price(symbol, side)
    if not entry_px:
        logger.warning("Entry price not found – skipping SL/TP.")
        return

    # Risk controls & TP grid
    set_two_tier_take_profits(symbol, side, qty, entry_px)
    set_trailing_stop(symbol, side, entry_px)
    set_hard_stop_loss(symbol, side, entry_px)

    log_trade(symbol, side, qty, entry_px)
    send_telegram(f"Opened {side} {symbol} @ {entry_px:.4f} – qty {qty}")


def close_position(symbol: str, side: str):
    pos_idx = 1 if side == "LONG" else 2
    trade_side = "Sell" if side == "LONG" else "Buy"
    resp = safe_api_call(session.get_positions, category="linear", symbol=symbol)
    if resp["retCode"] != 0:
        return
    size = 0.0
    for p in resp["result"]["list"]:
        if p["symbol"] == symbol and float(p["size"]) > 0 and ((side == "LONG" and p["side"] == "Buy") or (side == "SHORT" and p["side"] == "Sell")):
            size = float(p["size"])
            break
    if size == 0:
        logger.info(f"No {side} to close on {symbol}")
        return

    close = safe_api_call(session.place_order, category="linear", symbol=symbol,
                          side=trade_side, orderType="Market", qty=str(size),
                          timeInForce="GTC", positionIdx=pos_idx)
    if close["retCode"] != 0:
        logger.error(f"Close {side} error: {close}")
        return

    exit_px = get_current_price(symbol) or 0.0
    mark_trade_closed(symbol, side, exit_px)
    send_telegram(f"Closed {side} {symbol} @ {exit_px:.4f} – qty {size}")


# ---------------------------------------------------------------------------
#                   10)  Flask webhook – TradingView compatible
# ---------------------------------------------------------------------------

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    logger.debug(f"Webhook payload: {data}")
    if not data or "ticker" not in data or "direction" not in data:
        return jsonify({"error": "invalid payload"}), 400

    ticker = data["ticker"].upper()
    direction = data["direction"]

    try:
        if direction == "open_long":
            open_position(ticker, "BUY", "LONG")
        elif direction == "open_short":
            open_position(ticker, "SELL", "SHORT")
        elif direction == "close_long":
            close_position(ticker, "LONG")
        elif direction == "close_short":
            close_position(ticker, "SHORT")
        else:
            return jsonify({"error": "unknown direction"}), 400

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.exception("Webhook processing error")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
#                   11)  Misc helpers
# ---------------------------------------------------------------------------

def send_telegram(msg: str):
    try:
        bot.send_message(CHAT_ID, msg)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


# ---------------------------------------------------------------------------
#                   12)  Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_hedge_mode()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 80)))
