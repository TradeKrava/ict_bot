# =====================================================
# ICT SNIPER SCANNER — AI PRO EDITION
# Alpha Engine v4 (ICT 1:1) + Adaptive AI Risk Manager
# =====================================================

import os
import time
import math
import json
import csv
import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from http.server import HTTPServer, BaseHTTPRequestHandler

# =====================================================
# 🔑 TOKENS
# =====================================================

TG_TOKEN = os.getenv("TG_TOKEN")
if not TG_TOKEN:
    raise RuntimeError("Set TG_TOKEN env var")

# =====================================================
# 📊 BYBIT (USDT PERP)
# =====================================================

exchange = ccxt.bybit({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})

# =====================================================
# ⚙️ STATE
# =====================================================

STATE = {
    "running": False,
    "chat_id": None,

    "tfs": ["15m", "1h"],
    "universe": "top100",
    "whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],

    "only_close": True,
    "use_ema_trend": True,
    "use_ema_confirm": True,
    "use_pd": True,
    "use_atr": True,
    "use_manip": True,

    "pdLen": 50,
    "bosLookback": 5,
    "structLen": 3,

    "use_sltp": True,
    "slStyle": "Normal",
    "tpStyle": "Combined",
    "baseAtrMult": 0.6,

}

# =====================================================
# 🧮 INDICATORS
# =====================================================

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)

def atr(df, length):
    return true_range(df).ewm(span=length, adjust=False).mean()

# =====================================================
# 📦 DATA  ❗ FIXED
# =====================================================

async def fetch_ohlcv(symbol, tf, limit=240):
    data = await asyncio.to_thread(
        exchange.fetch_ohlcv,
        symbol,          # ✅ НЕ normalize_symbol
        tf,
        None,
        limit
    )
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def prepare_indicators(df):
    df = df.copy()
    df["ema9"]   = ema(df["close"], 9)
    df["ema21"]  = ema(df["close"], 21)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr14"]  = atr(df, 14)
    return df

# =====================================================
# 🧠 SIGNAL RESULT
# =====================================================

@dataclass
class SignalResult:
    symbol: str
    tf: str
    direction: str
    entry_style: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    info: dict

# =====================================================
# 🧠 FVG helpers ❗ FIXED
# =====================================================

def compute_last_fvg_levels(df):
    bull = bear = np.nan
    for i in range(2, len(df)):
        if df["high"].iloc[i-2] < df["low"].iloc[i]:
            bull = float(df["low"].iloc[i])
        if df["low"].iloc[i-2] > df["high"].iloc[i]:
            bear = float(df["high"].iloc[i])
    return bull, bear

def retest_bull_fvg(level, low, close):
    return not math.isnan(level) and low <= level and close > level

def retest_bear_fvg(level, high, close):
    return not math.isnan(level) and high >= level and close < level

# =====================================================
# 🧠 CORE ANALYZER (без змін)
# =====================================================

def analyze_symbol(df, symbol, tf):
    if len(df) < 220:
        return None

    idx = -2
    close = df["close"].iloc[idx]
    high  = df["high"].iloc[idx]
    low   = df["low"].iloc[idx]

    ema9   = df["ema9"].iloc[idx]
    ema21  = df["ema21"].iloc[idx]
    ema50  = df["ema50"].iloc[idx]
    ema200 = df["ema200"].iloc[idx]

    trendBull = ema50 > ema200 and close > ema50
    trendBear = ema50 < ema200 and close < ema50

    confirmBull = ema9 > ema21
    confirmBear = ema9 < ema21

    lastBullFVG, lastBearFVG = compute_last_fvg_levels(df.iloc[:idx])

    bullFVG = retest_bull_fvg(lastBullFVG, low, close)   # ✅ FIX
    bearFVG = retest_bear_fvg(lastBearFVG, high, close) # ✅ FIX

    if bullFVG and trendBull and confirmBull:
        direction = "BUY"
        style = "A — Conservative"
    elif bearFVG and trendBear and confirmBear:
        direction = "SELL"
        style = "A — Conservative"
    else:
        return None

    atr14 = df["atr14"].iloc[idx]
    entry = close
    sl = entry - atr14 if direction == "BUY" else entry + atr14
    tp1 = entry + atr14 if direction == "BUY" else entry - atr14
    tp2 = entry + atr14*2 if direction == "BUY" else entry - atr14*2
    tp3 = entry + atr14*3 if direction == "BUY" else entry - atr14*3

    return SignalResult(
        symbol=symbol,
        tf=tf,
        direction=direction,
        entry_style=style,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        info={}
    )

# =====================================================
# 🤖 TELEGRAM COMMANDS
# =====================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["chat_id"] = update.effective_chat.id
    await update.message.reply_text("✅ Бот готовий\n/startscan — старт\n/stopscan — стоп")

async def cmd_startscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["running"] = True
    STATE["chat_id"] = update.effective_chat.id
    await update.message.reply_text("▶️ Сканер запущено")

async def cmd_stopscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["running"] = False
    await update.message.reply_text("⏹ Сканер зупинено")

# =====================================================
# 🔁 SCANNER LOOP
# =====================================================

async def scanner_loop(app):
    while True:
        if not STATE["running"] or not STATE["chat_id"]:
            await asyncio.sleep(1)
            continue

        try:
            markets = await asyncio.to_thread(exchange.load_markets)
            symbols = [m for m in markets if markets[m].get("swap") and markets[m].get("quote") == "USDT"][:100]

            for s in symbols:
                for tf in STATE["tfs"]:
                    df = await fetch_ohlcv(s, tf)
                    df = prepare_indicators(df)
                    sig = analyze_symbol(df, s, tf)
                    if sig:
                        await app.bot.send_message(
                            chat_id=STATE["chat_id"],
                            text=(
                                f"🚨 {sig.direction} {s} {tf}\n"
                                f"Entry {sig.entry:.4f}\n"
                                f"SL {sig.sl:.4f}\n"
                                f"TP {sig.tp3:.4f}"
                            ),
                        )
                    await asyncio.sleep(0.3)   # ✅ пауза між TF / сигналами

                await asyncio.sleep(0.15)      # ✅ ГОЛОВНЕ: пауза між СИМВОЛАМИ

            await asyncio.sleep(3)
        except Exception as e:
            print("[scanner-error]", e)
            await asyncio.sleep(3)

# =====================================================
# 🤖 MAIN ❗ FIXED post_init
# =====================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def start_http():
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()

def main():
    async def post_init(application: Application):
        application.create_task(scanner_loop(application))
        print("[scanner] started")

    app = (
        Application.builder()
        .token(TG_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("startscan", cmd_startscan))
    app.add_handler(CommandHandler("stopscan", cmd_stopscan))

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    threading.Thread(target=start_http, daemon=True).start()
    main()
