from __future__ import annotations

"""
FULLA1 Alpaca trading bot (paper or live).

This bot mirrors the uploaded backtest strategy rules:
- 5-minute bars built from Alpaca 1-minute bars
- Premarket: 04:00-09:30 America/New_York
- Regular session entries: 09:30-15:30 America/New_York
- Premarket gain >= 3%
- Stock stronger than SPY, with SPY premarket gain <= 1%
- Premarket high is the key level
- At least 5 touches within 0.40%
- Last-30-minute premarket coil
- Strong green breakout candle, body >= 60%, upper wick <= 35%
- Breakout volume > previous rolling average (20 bars, minimum 3)
- Risk per trade = 10% of current equity
- Stop = key level * (1 - 0.5%)
- At +2%, sell 40%
- Remaining 60% keeps the original stop and exits at 15:55 ET

Trading mode is controlled by ALPACA_PAPER in the .env file.
Exit mode is controlled by USE_SCALE_OUT:
- true: sell 40% at TP1 and keep 60% with the original stop until 15:55 ET
- false: close 100% of the position at TP1 or the original stop
"""

import asyncio
import logging
import math
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)
from alpaca.trading.stream import TradingStream


# ========================= SETTINGS =========================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER = os.getenv("ALPACA_PAPER", "true").strip().lower() == "true"
USE_SCALE_OUT = os.getenv("USE_SCALE_OUT", "true").strip().lower() == "true"
DATA_FEED_NAME = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "Missing ALPACA_API_KEY or ALPACA_API_SECRET. Put them in a .env file."
    )

if DATA_FEED_NAME == "sip":
    DATA_FEED = DataFeed.SIP
elif DATA_FEED_NAME == "iex":
    DATA_FEED = DataFeed.IEX
else:
    raise ValueError("ALPACA_DATA_FEED must be 'iex' or 'sip'.")

TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "AMAT", "LRCX", "KLAC",
    "MU", "QCOM", "TXN", "ADI", "NXPI", "MRVL", "CRWD", "PANW", "FTNT",
    "SNPS", "CDNS", "ANET", "ORCL", "IBM", "CSCO", "ADBE", "CRM", "NOW",
    "PLTR", "SNOW", "SHOP",

    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",

    # Healthcare
    "LLY", "JNJ", "MRK", "ABBV", "ABT", "PFE", "BMY", "AMGN", "GILD", "ISRG",
    "MRNA", "VRTX",

    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "LOW", "MCD", "NKE", "SBUX", "BKNG", "RCL", "CCL",

    # Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP", "KHC", "MDLZ", "CL",

    # Energy
    "XOM", "CVX", "SLB", "HAL", "BKR", "DVN", "OXY", "COP", "EOG", "MPC",

    # Industrials
    "CAT", "DE", "GE", "HON", "ETN", "PH", "UPS", "UNP", "CSX", "LMT", "RTX",

    # Communications
    "META", "GOOGL", "NFLX", "DIS", "CMCSA", "TMUS", "T", "VZ",

    # Materials
    "LIN", "APD", "FCX", "NEM", "DOW",

    # Real Estate
    "PLD", "AMT", "EQIX",

    # High Beta / Momentum
    "HOOD", "SMCI", "ARM", "RDDT", "COIN", "NBIS", "APP", "RKLB", "HIMS", "CAVA",

    'SHPH','VMAR','MIMI','NVVE','SOBR','EHGO','SKYQ','VEEE',
]

ALL_SYMBOLS = sorted(set(TICKERS + ["SPY"]))

# The Alpaca Basic plan supports 30 stock websocket subscriptions.
# This exact strategy uses more symbols and therefore requires SIP / Algo Trader Plus.
if DATA_FEED_NAME == "iex" and len(ALL_SYMBOLS) > 30:
    raise RuntimeError(
        f"ALPACA_DATA_FEED=iex supports only 30 websocket symbols on the Basic plan, "
        f"but this bot requires {len(ALL_SYMBOLS)} symbols including SPY. "
        "Set ALPACA_DATA_FEED=sip with an eligible Alpaca market-data subscription, "
        "or reduce the ticker universe to at most 29 stocks plus SPY."
    )

TIMEFRAME_MINUTES = 5
RISK_PER_TRADE = 0.10

MIN_PREMARKET_GAIN = 0.03
MAX_SPY_GAIN_FOR_RS = 0.01
TOUCH_TOLERANCE_PCT = 0.40
MIN_TOUCHES = 5

COIL_MINUTES_BEFORE_OPEN = 30
COIL_TOLERANCE_MULT = 3.0

BODY_STRENGTH_MIN = 0.60
UPPER_WICK_MAX = 0.35
VOLUME_LOOKBACK = 20

FIRST_TARGET_PCT = 0.02
SCALE_OUT_PCT = 0.40
STOP_BUFFER_PCT = 0.005

ENTRY_START = dtime(9, 30)
ENTRY_END = dtime(15, 30)
CLOSE_ALL_TIME = dtime(15, 55)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

ORDER_FILL_TIMEOUT_SECONDS = 30
POLL_SECONDS = 0.5
MAX_CACHE_ROWS_PER_SYMBOL = 2000


# ========================= LOGGING =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FULLA1")


# ========================= CLIENTS =========================

trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
history_client = StockHistoricalDataClient(API_KEY, API_SECRET)
data_stream = StockDataStream(API_KEY, API_SECRET, feed=DATA_FEED)
trading_stream = TradingStream(API_KEY, API_SECRET, paper=PAPER)


# ========================= STATE =========================

@dataclass
class PositionState:
    symbol: str
    signal_time: pd.Timestamp
    signal_price: float
    key_level: float
    stop_price: float
    first_target: float
    requested_qty: int
    filled_qty: int = 0
    actual_entry_price: float = 0.0
    scaled_qty: int = 0
    runner_qty: int = 0
    entry_order_id: Optional[str] = None
    scale_oco_order_id: Optional[str] = None
    runner_stop_order_id: Optional[str] = None
    active: bool = False
    scaled: bool = False


bars_cache: dict[str, pd.DataFrame] = {
    symbol: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    for symbol in ALL_SYMBOLS
}
processed_5m_buckets: dict[str, set[pd.Timestamp]] = {symbol: set() for symbol in ALL_SYMBOLS}
traded_dates: dict[str, set[str]] = {symbol: set() for symbol in TICKERS}
position_states: dict[str, PositionState] = {}
entry_in_progress: set[str] = set()
state_lock = threading.RLock()


# ========================= DATA HELPERS =========================

def _ensure_ny_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def append_minute_bar(symbol: str, timestamp, open_, high, low, close, volume) -> None:
    ts = _ensure_ny_timestamp(timestamp)
    row = pd.DataFrame(
        [{
            "timestamp": ts,
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }]
    )

    with state_lock:
        df = pd.concat([bars_cache[symbol], row], ignore_index=True)
        df = (
            df.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(MAX_CACHE_ROWS_PER_SYMBOL)
            .reset_index(drop=True)
        )
        bars_cache[symbol] = df


def build_5m_bars(symbol: str) -> pd.DataFrame:
    with state_lock:
        minute_df = bars_cache[symbol].copy()

    if minute_df.empty:
        return pd.DataFrame()

    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"])
    minute_df = minute_df.set_index("timestamp").sort_index()

    bars_5m = minute_df.resample(
        f"{TIMEFRAME_MINUTES}min",
        label="left",
        closed="left",
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_count=("close", "count"),
    )

    bars_5m = bars_5m.dropna(subset=["open", "high", "low", "close"])
    bars_5m = bars_5m.reset_index()
    bars_5m["symbol"] = symbol
    bars_5m["date"] = bars_5m["timestamp"].dt.date.astype(str)
    bars_5m["tod"] = bars_5m["timestamp"].dt.time
    bars_5m["is_premarket"] = (
        (bars_5m["tod"] >= dtime(4, 0)) & (bars_5m["tod"] < dtime(9, 30))
    )
    bars_5m["is_regular"] = (
        (bars_5m["tod"] >= dtime(9, 30)) & (bars_5m["tod"] <= dtime(16, 0))
    )
    return bars_5m


def premarket_gain(premarket: pd.DataFrame) -> Optional[float]:
    if premarket.empty:
        return None
    first_open = float(premarket.iloc[0]["open"])
    last_close = float(premarket.iloc[-1]["close"])
    if first_open <= 0:
        return None
    return (last_close - first_open) / first_open


def level_touch_count(premarket: pd.DataFrame, key_level: float) -> int:
    tolerance = key_level * TOUCH_TOLERANCE_PCT / 100
    touches = (
        (premarket["high"].sub(key_level).abs() <= tolerance)
        & (premarket["close"] < key_level)
    )
    return int(touches.sum())


def respected_level_in_last_30min(premarket: pd.DataFrame, key_level: float) -> bool:
    if premarket.empty:
        return False

    cutoff = pd.Timestamp(premarket.iloc[-1]["timestamp"]) - pd.Timedelta(
        minutes=COIL_MINUTES_BEFORE_OPEN
    )
    recent = premarket[premarket["timestamp"] >= cutoff]
    if len(recent) < 2:
        return False

    tolerance = (
        key_level * TOUCH_TOLERANCE_PCT / 100 * COIL_TOLERANCE_MULT
    )
    near_level = (
        (recent["close"] < key_level)
        & (recent["close"] >= key_level - tolerance)
    )
    no_clean_break_preopen = (
        recent["close"].max()
        <= key_level + (key_level * TOUCH_TOLERANCE_PCT / 100)
    )
    return bool(near_level.mean() >= 0.50 and no_clean_break_preopen)


def breakout_bar_ok(bar: pd.Series, key_level: float, avg_volume: float) -> bool:
    candle_range = float(bar["high"] - bar["low"])
    if candle_range <= 0:
        return False

    body = abs(float(bar["close"] - bar["open"]))
    body_pct = body / candle_range
    upper_wick = float(bar["high"] - max(bar["open"], bar["close"]))
    upper_wick_pct = upper_wick / candle_range

    closes_above = float(bar["close"]) > key_level
    opened_at_or_below = float(bar["open"]) <= key_level
    green = float(bar["close"]) > float(bar["open"])
    increased_volume = (
        float(bar["volume"]) > avg_volume
        if avg_volume and not np.isnan(avg_volume)
        else False
    )

    return bool(
        closes_above
        and opened_at_or_below
        and green
        and body_pct >= BODY_STRENGTH_MIN
        and upper_wick_pct <= UPPER_WICK_MAX
        and increased_volume
    )


# ========================= ORDER HELPERS =========================

def wait_for_terminal_order(order_id: str, timeout: int = ORDER_FILL_TIMEOUT_SECONDS):
    deadline = time.time() + timeout
    while time.time() < deadline:
        order = trading_client.get_order_by_id(order_id)
        status = order.status
        if status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }:
            return order
        time.sleep(POLL_SECONDS)
    return trading_client.get_order_by_id(order_id)


def cancel_symbol_orders(symbol: str) -> None:
    try:
        orders = trading_client.get_orders()
        for order in orders:
            if order.symbol == symbol and order.status in {
                OrderStatus.NEW,
                OrderStatus.ACCEPTED,
                OrderStatus.PENDING_NEW,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.HELD,
                OrderStatus.CALCULATED,
            }:
                try:
                    trading_client.cancel_order_by_id(order.id)
                except Exception as exc:
                    logger.warning("Could not cancel %s order %s: %s", symbol, order.id, exc)
    except Exception as exc:
        logger.error("Failed to query/cancel orders for %s: %s", symbol, exc)


def place_exit_orders(state: PositionState) -> None:
    """Place broker-managed exit orders.

    USE_SCALE_OUT = True:
      - 40% OCO: TP1 at +2% or original stop
      - 60%: independent original stop, then EOD close at 15:55 ET

    USE_SCALE_OUT = False:
      - 100% OCO: TP1 at +2% or original stop
      - no runner and no partial profit-taking
    """
    if USE_SCALE_OUT:
        if state.scaled_qty > 0:
            scale_order = LimitOrderRequest(
                symbol=state.symbol,
                qty=state.scaled_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(state.first_target, 2),
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=round(state.first_target, 2)),
                stop_loss=StopLossRequest(stop_price=round(state.stop_price, 2)),
            )
            scale_parent = trading_client.submit_order(scale_order)
            state.scale_oco_order_id = str(scale_parent.id)

        if state.runner_qty > 0:
            runner_stop = StopOrderRequest(
                symbol=state.symbol,
                qty=state.runner_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(state.stop_price, 2),
            )
            runner_order = trading_client.submit_order(runner_stop)
            state.runner_stop_order_id = str(runner_order.id)
    else:
        full_exit = LimitOrderRequest(
            symbol=state.symbol,
            qty=state.filled_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(state.first_target, 2),
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=round(state.first_target, 2)),
            stop_loss=StopLossRequest(stop_price=round(state.stop_price, 2)),
        )
        full_exit_parent = trading_client.submit_order(full_exit)
        state.scale_oco_order_id = str(full_exit_parent.id)


async def execute_entry(
    symbol: str,
    signal_time: pd.Timestamp,
    signal_price: float,
    key_level: float,
    touches: int,
    stock_pm_gain: float,
    spy_pm_gain: float,
) -> None:
    with state_lock:
        if symbol in entry_in_progress or symbol in position_states:
            return
        entry_in_progress.add(symbol)

    try:
        account = await asyncio.to_thread(trading_client.get_account)
        equity = float(account.equity)

        stop_price = key_level * (1 - STOP_BUFFER_PCT)
        risk_per_share = signal_price - stop_price
        if risk_per_share <= 0:
            logger.info("%s rejected: non-positive risk per share", symbol)
            return

        dollars_risked = equity * RISK_PER_TRADE
        qty = int(dollars_risked / risk_per_share)
        if qty <= 0:
            logger.info("%s rejected: calculated quantity is zero", symbol)
            return

        first_target = signal_price * (1 + FIRST_TARGET_PCT)

        entry_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"FULLA1-{symbol}-{signal_time.strftime('%Y%m%d-%H%M')}",
        )

        logger.info(
            "SIGNAL %s | time=%s price=%.4f key=%.4f stop=%.4f target=%.4f "
            "qty=%d pm=%.2f%% spy=%.2f%% touches=%d",
            symbol,
            signal_time,
            signal_price,
            key_level,
            stop_price,
            first_target,
            qty,
            stock_pm_gain * 100,
            spy_pm_gain * 100,
            touches,
        )

        submitted = await asyncio.to_thread(trading_client.submit_order, entry_request)
        filled_order = await asyncio.to_thread(wait_for_terminal_order, str(submitted.id))

        if filled_order.status != OrderStatus.FILLED:
            logger.error(
                "%s entry did not fill. status=%s reason=%s",
                symbol,
                filled_order.status,
                getattr(filled_order, "reject_reason", None),
            )
            return

        filled_qty = int(float(filled_order.filled_qty))
        actual_entry_price = float(filled_order.filled_avg_price)
        if USE_SCALE_OUT:
            scaled_qty = int(filled_qty * SCALE_OUT_PCT)
            runner_qty = filled_qty - scaled_qty
        else:
            scaled_qty = filled_qty
            runner_qty = 0

        state = PositionState(
            symbol=symbol,
            signal_time=signal_time,
            signal_price=signal_price,
            key_level=key_level,
            stop_price=stop_price,
            first_target=first_target,
            requested_qty=qty,
            filled_qty=filled_qty,
            actual_entry_price=actual_entry_price,
            scaled_qty=scaled_qty,
            runner_qty=runner_qty,
            entry_order_id=str(filled_order.id),
            active=True,
        )

        await asyncio.to_thread(place_exit_orders, state)

        with state_lock:
            position_states[symbol] = state
            traded_dates[symbol].add(signal_time.date().isoformat())

        logger.info(
            "ENTERED %s | filled=%d avg=%.4f | exit_mode=%s | TP1 qty=%d | runner qty=%d",
            symbol,
            filled_qty,
            actual_entry_price,
            "40% scale + 60% runner" if USE_SCALE_OUT else "100% exit at TP1",
            scaled_qty,
            runner_qty,
        )

    except Exception as exc:
        logger.exception("Entry workflow failed for %s: %s", symbol, exc)
        await asyncio.to_thread(cancel_symbol_orders, symbol)
    finally:
        with state_lock:
            entry_in_progress.discard(symbol)


# ========================= STRATEGY EVALUATION =========================

async def evaluate_completed_5m_bar(symbol: str, bar_time: pd.Timestamp) -> None:
    if symbol == "SPY":
        return

    date_str = bar_time.date().isoformat()
    bar_tod = bar_time.time()

    if bar_tod < ENTRY_START or bar_tod > ENTRY_END:
        return

    with state_lock:
        if date_str in traded_dates[symbol]:
            return
        if symbol in position_states or symbol in entry_in_progress:
            return

    symbol_5m = build_5m_bars(symbol)
    spy_5m = build_5m_bars("SPY")
    if symbol_5m.empty or spy_5m.empty:
        return

    symbol_day = symbol_5m[symbol_5m["date"] == date_str].copy()
    spy_day = spy_5m[spy_5m["date"] == date_str].copy()

    pre = symbol_day[symbol_day["is_premarket"]].copy()
    regular = symbol_day[symbol_day["is_regular"]].copy().reset_index(drop=True)
    spy_pre = spy_day[spy_day["is_premarket"]].copy()

    if pre.empty or regular.empty or spy_pre.empty:
        return

    stock_pm_gain = premarket_gain(pre)
    spy_pm_gain = premarket_gain(spy_pre)
    if stock_pm_gain is None or spy_pm_gain is None:
        return

    clear_bullish_momentum = (
        stock_pm_gain >= MIN_PREMARKET_GAIN
        and stock_pm_gain > spy_pm_gain
        and spy_pm_gain <= MAX_SPY_GAIN_FOR_RS
    )
    if not clear_bullish_momentum:
        return

    key_level = float(pre["high"].max())
    touches = level_touch_count(pre, key_level)
    if touches < MIN_TOUCHES:
        return

    if not respected_level_in_last_30min(pre, key_level):
        return

    regular["avg_volume"] = (
        regular["volume"]
        .rolling(VOLUME_LOOKBACK, min_periods=3)
        .mean()
        .shift(1)
    )

    matching = regular[regular["timestamp"] == bar_time]
    if matching.empty:
        return

    bar = matching.iloc[-1]
    avg_volume = float(bar["avg_volume"]) if not pd.isna(bar["avg_volume"]) else np.nan

    if breakout_bar_ok(bar, key_level, avg_volume):
        asyncio.create_task(
            execute_entry(
                symbol=symbol,
                signal_time=bar_time,
                signal_price=float(bar["close"]),
                key_level=key_level,
                touches=touches,
                stock_pm_gain=stock_pm_gain,
                spy_pm_gain=spy_pm_gain,
            )
        )


async def on_minute_bar(bar) -> None:
    symbol = bar.symbol
    if symbol not in bars_cache:
        return

    append_minute_bar(
        symbol=symbol,
        timestamp=bar.timestamp,
        open_=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )

    minute_ts = _ensure_ny_timestamp(bar.timestamp)

    # A bar stamped HH:04 completes the HH:00-HH:05 five-minute bucket.
    if minute_ts.minute % TIMEFRAME_MINUTES != TIMEFRAME_MINUTES - 1:
        return

    bucket = minute_ts.floor(f"{TIMEFRAME_MINUTES}min")
    with state_lock:
        if bucket in processed_5m_buckets[symbol]:
            return
        processed_5m_buckets[symbol].add(bucket)

    await evaluate_completed_5m_bar(symbol, bucket)


async def on_trade_update(update) -> None:
    try:
        event = str(update.event)
        order = update.order
        symbol = order.symbol
        logger.info(
            "ORDER UPDATE | %s | %s | id=%s status=%s filled=%s avg=%s",
            symbol,
            event,
            order.id,
            order.status,
            order.filled_qty,
            order.filled_avg_price,
        )

        # Remove state once the broker reports there is no position.
        if event in {"fill", "canceled", "rejected", "expired"} and symbol in position_states:
            try:
                await asyncio.to_thread(trading_client.get_open_position, symbol)
            except Exception:
                with state_lock:
                    position_states.pop(symbol, None)
                logger.info("%s position is closed; local state cleared", symbol)
    except Exception as exc:
        logger.exception("Trade-update handler error: %s", exc)


# ========================= WARM-UP / SAFETY =========================

def recover_existing_state() -> None:
    """Recover Alpaca-held positions and orders after a Render restart.

    Existing broker exit orders are preserved. Recovered symbols are marked as
    already traded today, preventing duplicate entries. Open orders that have no
    corresponding position are canceled as stale.
    """
    positions = trading_client.get_all_positions()
    open_orders = trading_client.get_orders()
    today = datetime.now(NY).date().isoformat()
    position_symbols = {p.symbol for p in positions}

    for position in positions:
        symbol = position.symbol
        qty = int(abs(float(position.qty)))
        avg_entry = float(position.avg_entry_price)
        position_states[symbol] = PositionState(
            symbol=symbol,
            signal_time=pd.Timestamp.now(tz="America/New_York"),
            signal_price=avg_entry,
            key_level=avg_entry,
            stop_price=0.0,
            first_target=0.0,
            requested_qty=qty,
            filled_qty=qty,
            actual_entry_price=avg_entry,
            runner_qty=qty,
            active=True,
        )
        if symbol in traded_dates:
            traded_dates[symbol].add(today)
        logger.warning(
            "RECOVERED POSITION %s | qty=%d avg_entry=%.4f | broker orders preserved",
            symbol, qty, avg_entry,
        )

    for order in open_orders:
        if order.symbol not in position_symbols:
            try:
                trading_client.cancel_order_by_id(order.id)
                logger.warning(
                    "Canceled stale order %s for %s because no open position exists",
                    order.id, order.symbol,
                )
            except Exception as exc:
                logger.warning("Could not cancel stale order %s: %s", order.id, exc)

    logger.info(
        "Recovery complete | positions=%d open_orders=%d",
        len(positions), len(open_orders),
    )


def warmup_today() -> None:
    now_ny = datetime.now(NY)
    start_ny = datetime.combine(
        now_ny.date(),
        dtime(4, 0),
        tzinfo=NY,
    )

    if now_ny < start_ny:
        logger.info("Before 04:00 ET; warm-up skipped")
        return

    request = StockBarsRequest(
        symbol_or_symbols=ALL_SYMBOLS,
        timeframe=TimeFrame.Minute,
        start=start_ny.astimezone(UTC),
        end=now_ny.astimezone(UTC),
        feed=DATA_FEED,
    )

    logger.info(
        "Warm-up: loading today's minute bars for %d symbols",
        len(ALL_SYMBOLS),
    )

    result = history_client.get_stock_bars(request).df

    if result.empty:
        logger.warning("Warm-up returned no data")
        return

    for symbol in ALL_SYMBOLS:
        try:
            symbol_df = result.loc[symbol].reset_index()
        except KeyError:
            continue

        if symbol_df.empty:
            continue

        symbol_df["timestamp"] = (
            pd.to_datetime(symbol_df["timestamp"])
            .dt.tz_convert("America/New_York")
        )

        bars_cache[symbol] = (
            symbol_df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .tail(MAX_CACHE_ROWS_PER_SYMBOL)
            .reset_index(drop=True)
        )

        bars_5m = build_5m_bars(symbol)

        if bars_5m.empty:
            continue

        # Premarket candles are historical context only.
        # Mark them processed so they are never treated as entry signals.
        premarket_buckets = bars_5m.loc[
            bars_5m["is_premarket"],
            "timestamp",
        ].tolist()

        processed_5m_buckets[symbol].update(
            premarket_buckets
        )

    logger.info(
        "Warm-up complete: premarket history loaded from 04:00 ET"
    )

async def evaluate_latest_completed_bars() -> None:
    """
    Checks the latest completed 5-minute regular-session bar
    after the historical warm-up.

    This allows the bot to start after 09:30 ET without ignoring
    the most recent completed breakout candle.
    """

    now_ny = pd.Timestamp.now(tz="America/New_York")

    # Current bucket is still forming, so only evaluate bars
    # whose five-minute period has fully completed.
    current_bucket = now_ny.floor(
        f"{TIMEFRAME_MINUTES}min"
    )

    for symbol in TICKERS:
        bars_5m = build_5m_bars(symbol)

        if bars_5m.empty:
            continue

        completed_regular = bars_5m[
            (bars_5m["is_regular"])
            & (bars_5m["timestamp"] < current_bucket)
        ].copy()

        if completed_regular.empty:
            continue

        # Only check the latest completed candle.
        latest_bar_time = completed_regular.iloc[-1][
            "timestamp"
        ]

        if (
            latest_bar_time.time() < ENTRY_START
            or latest_bar_time.time() > ENTRY_END
        ):
            continue

        with state_lock:
            if latest_bar_time in processed_5m_buckets[symbol]:
                continue

            processed_5m_buckets[symbol].add(
                latest_bar_time
            )

        await evaluate_completed_5m_bar(
            symbol,
            latest_bar_time,
        )

# ========================= END-OF-DAY MANAGER =========================

async def close_symbol_at_eod(symbol: str) -> None:
    logger.info("EOD close starting for %s", symbol)
    await asyncio.to_thread(cancel_symbol_orders, symbol)
    await asyncio.sleep(1.0)

    try:
        await asyncio.to_thread(trading_client.close_position, symbol)
        logger.info("EOD market close submitted for %s", symbol)
    except Exception as exc:
        logger.warning("EOD close for %s returned: %s", symbol, exc)

    with state_lock:
        position_states.pop(symbol, None)


async def eod_manager() -> None:
    last_eod_date: Optional[str] = None

    while True:
        now_ny = datetime.now(NY)
        today = now_ny.date().isoformat()

        if now_ny.time() >= CLOSE_ALL_TIME and last_eod_date != today:
            with state_lock:
                symbols = list(position_states.keys())

            for symbol in symbols:
                await close_symbol_at_eod(symbol)

            last_eod_date = today

        # Reset daily local tracking shortly after midnight ET.
        if now_ny.time() < dtime(0, 5):
            with state_lock:
                for symbol in TICKERS:
                    traded_dates[symbol] = {
                        d for d in traded_dates[symbol] if d == today
                    }
                    processed_5m_buckets[symbol].clear()
                processed_5m_buckets["SPY"].clear()

        await asyncio.sleep(1.0)


# ========================= RUN =========================

def start_streams() -> None:
    data_stream.subscribe_bars(on_minute_bar, *ALL_SYMBOLS)
    trading_stream.subscribe_trade_updates(on_trade_update)

    data_thread = threading.Thread(
        target=data_stream.run,
        name="alpaca-market-data",
        daemon=True,
    )
    trading_thread = threading.Thread(
        target=trading_stream.run,
        name="alpaca-trading-updates",
        daemon=True,
    )

    data_thread.start()
    trading_thread.start()


async def shutdown_streams() -> None:
    logger.info("Stopping Alpaca streams")
    for stream_name, stream in (("market-data", data_stream), ("trade-updates", trading_stream)):
        try:
            result = stream.stop()
            if asyncio.iscoroutine(result):
                await result
            logger.info("Stopped %s stream", stream_name)
        except Exception as exc:
            logger.warning("Could not stop %s stream cleanly: %s", stream_name, exc)


async def main() -> None:
    logger.info(
        "Starting FULLA1 bot | paper=%s feed=%s symbols=%d risk=%.1f%% "
        "exit_mode=%s render_instance=%s",
        PAPER,
        DATA_FEED_NAME,
        len(TICKERS),
        RISK_PER_TRADE * 100,
        "40% scale + 60% runner" if USE_SCALE_OUT else "100% exit at TP1",
        os.getenv("RENDER_INSTANCE_ID", "local"),
    )

    await asyncio.to_thread(recover_existing_state)
    await asyncio.to_thread(warmup_today)
    await evaluate_latest_completed_bars()
    start_streams()

    try:
        await eod_manager()
    finally:
        await shutdown_streams()


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    run()
