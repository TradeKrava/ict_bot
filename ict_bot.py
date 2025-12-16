"""
Alpha Engine v4 — PRO ICT Sniper 2025 (Telegram Bot)
- Bybit USDT Perpetual (ccxt)
- python-telegram-bot v21+

ВАЖЛИВО:
  1) Set env var TG_TOKEN with your bot token (НЕ хардкодь токен в коді).
  2) (Опційно) BYBIT_API_KEY / BYBIT_API_SECRET якщо буде жорсткий rate limit (публічні дані теж працюють).

Run:
  pip install python-telegram-bot==21.* ccxt pandas numpy
  python bot.py
"""

from __future__ import annotations

import os
import math
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

# ==============================
# 🔑 TELEGRAM TOKEN
# ==============================
TG_TOKEN = os.getenv("TG_TOKEN")
if not TG_TOKEN:
    raise RuntimeError("Set TG_TOKEN env var (do NOT hardcode your token).")

# ==============================
# 📊 BYBIT (USDT PERP)
# ==============================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"
    }
})
async def load_all_usdt_perp_symbols() -> tuple[str, ...]:
    def _load():
        exchange.load_markets()
        symbols = []
        for m in exchange.markets.values():
            if (
                m.get("quote") == "USDT"
                and m.get("swap") is True
                and m.get("active") is True
            ):
                symbols.append(m["symbol"].replace("/", "").replace(":USDT", ""))
        return tuple(sorted(set(symbols)))

    return await asyncio.to_thread(_load)
  
# ==============================
# ✅ CONFIG (як у твоєму Pine)
# ==============================

@dataclass
class SniperConfig:
    # BASIC INPUTS
    onlyClose: bool = True
    useTrend: bool = True                 # legacy, залишив для паритету
    usePD: bool = True
    useLiq: bool = True                   # legacy, sweep рахується
    useFVG: bool = True
    useATR: bool = True

    pdLen: int = 50
    bosLookback: int = 5
    structLen: int = 3

    # EMA FILTERS
    useEMAtrend: bool = True              # 🔥 EMA 50/200 тренд
    useEMAconfirm: bool = True            # 🔥 EMA 9/21 підтвердження

    # MANIPULATIONS
    useManip: bool = True
    useManipLabels: bool = True           # для бота: писати в тексті

    # SL/TP
    useSLA: bool = True
    slStyle: str = "Normal"               # Aggressive/Normal/Safe
    tpStyle: str = "Combined"             # R-multiple/Structural/Combined
    usePartial: bool = True
    baseAtrMult: float = 0.6

    # Bot specifics
    timeframe: str = "15m"
    scan_interval_sec: int = 60
    symbols: Tuple[str, ...] = ()
    universe: str = "all"   # all | custom


STATE: Dict[str, object] = {
    "running": False,
    "chat_id": None,
    "task": None,
    "last_signal": {},   # key=(symbol, tf) -> {"ts": int, "side": "BUY"/"SELL"}
    "cfg": SniperConfig(),
}

# ==============================
# 🧮 INDICATORS
# ==============================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    n = len(high)
    out = pd.Series([np.nan] * n, index=high.index, dtype="float64")
    arr = high.to_numpy()
    for i in range(left, n - right):
        win = arr[i - left : i + right + 1]
        if np.isnan(arr[i]):
            continue
        if arr[i] == np.nanmax(win) and (win == arr[i]).sum() == 1:
            out.iat[i] = arr[i]
    return out

def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = pd.Series([np.nan] * n, index=low.index, dtype="float64")
    arr = low.to_numpy()
    for i in range(left, n - right):
        win = arr[i - left : i + right + 1]
        if np.isnan(arr[i]):
            continue
        if arr[i] == np.nanmin(win) and (win == arr[i]).sum() == 1:
            out.iat[i] = arr[i]
    return out

def safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

# ==============================
# 📈 CORE LOGIC (Pine → Python)
# ==============================

@dataclass
class SignalResult:
    bull: bool
    bear: bool
    style: str
    entry: Optional[float]
    sl: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    deal_status: str
    extra: Dict[str, str]

def compute_signal(df: pd.DataFrame, cfg: SniperConfig) -> SignalResult:
    if len(df) < max(250, cfg.pdLen + 10, cfg.structLen * 4, cfg.bosLookback + 10):
        return SignalResult(False, False, "—", None, None, None, None, None, "Недостатньо даних", {})

    idx = -2 if cfg.onlyClose else -1  # як barstate.isconfirmed
    dfv = df.copy()

    # PD zone
    sHigh = dfv["high"].rolling(cfg.pdLen).max()
    sLow = dfv["low"].rolling(cfg.pdLen).min()
    midPD = (sHigh + sLow) / 2.0

    inDiscount = dfv["close"] < midPD
    inPremium = dfv["close"] > midPD

    # Sweeps
    liqLow = dfv["low"] < dfv["low"].shift(1).rolling(5).min()
    liqHigh = dfv["high"] > dfv["high"].shift(1).rolling(5).max()

    # ATR candle filter: (high-low) >= ATR14*0.6
    atr14 = atr(dfv, 14)
    atrPass = (~cfg.useATR) | ((dfv["high"] - dfv["low"]) >= (atr14 * 0.6))

    # FVG
    bullFVG = dfv["high"].shift(2) < dfv["low"]
    bearFVG = dfv["low"].shift(2) > dfv["high"]

    lastBullFVG = pd.Series(np.nan, index=dfv.index, dtype="float64")
    lastBearFVG = pd.Series(np.nan, index=dfv.index, dtype="float64")

    cur_bull = np.nan
    cur_bear = np.nan
    for i in range(len(dfv)):
        if bool(bullFVG.iat[i]):
            cur_bull = dfv["low"].iat[i]
        if bool(bearFVG.iat[i]):
            cur_bear = dfv["high"].iat[i]
        lastBullFVG.iat[i] = cur_bull
        lastBearFVG.iat[i] = cur_bear

    # EMA
    ema9 = ema(dfv["close"], 9)
    ema21 = ema(dfv["close"], 21)
    ema50 = ema(dfv["close"], 50)
    ema200 = ema(dfv["close"], 200)

    trendBull = (ema50 > ema200) & (dfv["close"] > ema50)
    trendBear = (ema50 < ema200) & (dfv["close"] < ema50)

    confirmBull = ema9 > ema21
    confirmBear = ema9 < ema21

    finalBull_OK = ((~cfg.useEMAtrend) | trendBull) & ((~cfg.useEMAconfirm) | confirmBull)
    finalBear_OK = ((~cfg.useEMAtrend) | trendBear) & ((~cfg.useEMAconfirm) | confirmBear)

    pullbackEMA21_bull = (dfv["low"] <= ema21) & (dfv["close"] > ema21)
    pullbackEMA21_bear = (dfv["high"] >= ema21) & (dfv["close"] < ema21)

    # Structure pivots
    ph = pivot_high(dfv["high"], cfg.structLen, cfg.structLen)
    pl = pivot_low(dfv["low"], cfg.structLen, cfg.structLen)

    lastHigh = np.nan
    prevHigh = np.nan
    lastLow = np.nan
    prevLow = np.nan
    for i in range(len(dfv)):
        if not math.isnan(ph.iat[i]):
            prevHigh, lastHigh = lastHigh, ph.iat[i]
        if not math.isnan(pl.iat[i]):
            prevLow, lastLow = lastLow, pl.iat[i]

    hasStruct = not any(map(lambda x: (x is None) or math.isnan(x), [lastHigh, prevHigh, lastLow, prevLow]))
    structUp = hasStruct and (lastHigh > prevHigh) and (lastLow > prevLow)
    structDown = hasStruct and (lastHigh < prevHigh) and (lastLow < prevLow)

    close_i = float(dfv["close"].iat[idx])
    low_i = float(dfv["low"].iat[idx])
    high_i = float(dfv["high"].iat[idx])

    isHL = hasStruct and (lastLow > prevLow)
    isLH = hasStruct and (lastHigh < prevHigh)

    # Manipulations SPRING/UTAD
    sweepLow = bool(liqLow.iat[idx])
    sweepHigh = bool(liqHigh.iat[idx])

    bosUpAfterSweep = sweepLow and hasStruct and (close_i > lastHigh)
    bosDownAfterSweep = sweepHigh and hasStruct and (close_i < lastLow)

    manipBuy = cfg.useManip and bosUpAfterSweep
    manipSell = cfg.useManip and bosDownAfterSweep

    # ICT BOS (lookback)
    hi_lb = dfv["high"].shift(1).rolling(cfg.bosLookback).max()
    lo_lb = dfv["low"].shift(1).rolling(cfg.bosLookback).min()

    bosUpIct = bool(liqLow.iat[idx]) and (close_i > float(hi_lb.iat[idx]))
    bosDownIct = bool(liqHigh.iat[idx]) and (close_i < float(lo_lb.iat[idx]))

    # FVG retest
    lastBear = safe_float(lastBearFVG.iat[idx])
    lastBull = safe_float(lastBullFVG.iat[idx])

    bullRetestFVG_A = (lastBear is not None) and (low_i <= lastBear) and (close_i > lastBear)
    bearRetestFVG_A = (lastBull is not None) and (high_i >= lastBull) and (close_i < lastBull)

    inDisc_i = bool(inDiscount.iat[idx])
    inPrem_i = bool(inPremium.iat[idx])
    atrPass_i = bool(atrPass.iat[idx])

    finalBull_i = bool(finalBull_OK.iat[idx])
    finalBear_i = bool(finalBear_OK.iat[idx])

    # A/B/C
    bullA = bosUpIct and (bullRetestFVG_A or bool(pullbackEMA21_bull.iat[idx])) and ((not cfg.usePD) or inDisc_i) and finalBull_i and atrPass_i
    bearA = bosDownIct and (bearRetestFVG_A or bool(pullbackEMA21_bear.iat[idx])) and ((not cfg.usePD) or inPrem_i) and finalBear_i and atrPass_i

    bullB = bosUpIct and isHL and finalBull_i and atrPass_i
    bearB = bosDownIct and isLH and finalBear_i and atrPass_i

    bullC = bosUpIct and bool(trendBull.iat[idx]) and atrPass_i
    bearC = bosDownIct and bool(trendBear.iat[idx]) and atrPass_i

    bullSignal = False
    bearSignal = False
    entryStyle = "—"

    if bullA:
        bullSignal, entryStyle = True, "A — Conservative"
    elif bullB:
        bullSignal, entryStyle = True, "B — Normal"
    elif bullC:
        bullSignal, entryStyle = True, "C — Aggressive"
    elif bearA:
        bearSignal, entryStyle = True, "A — Conservative"
    elif bearB:
        bearSignal, entryStyle = True, "B — Normal"
    elif bearC:
        bearSignal, entryStyle = True, "C — Aggressive"

    entry = close_i if (bullSignal or bearSignal) else None

    # ===== SL/TP Engine (твоя логіка) =====
    sl = tp1 = tp2 = tp3 = None
    deal_status = "Немає активної угоди"

    if cfg.useSLA and entry is not None and hasStruct:
        style_mult = 0.5 if cfg.slStyle == "Aggressive" else 1.0 if cfg.slStyle == "Normal" else 1.5

        dyn_mult = 1.0
        if manipBuy or manipSell:
            dyn_mult += 0.4

        dist21 = abs(entry - float(ema21.iat[idx])) / entry
        dist50 = abs(entry - float(ema50.iat[idx])) / entry
        if dist21 < 0.004:
            dyn_mult -= 0.25
        elif dist50 < 0.006:
            dyn_mult -= 0.15
        dyn_mult = max(0.5, min(dyn_mult, 2.0))

        slAtrMult = cfg.baseAtrMult * style_mult * dyn_mult
        atr5 = atr(dfv, 5)
        slAtr = (float(atr5.iat[idx]) if not math.isnan(float(atr5.iat[idx])) else float(atr14.iat[idx])) * slAtrMult

        ch50 = (ema50.iat[idx] - ema50.iat[idx - 5]) / ema50.iat[idx] * 100 if idx - 5 >= -len(dfv) else 0.0

        if bullSignal:
            refLow1 = low_i
            refLow2 = lastLow
            rawSL = refLow1 if cfg.slStyle == "Aggressive" else min(refLow1, refLow2) if cfg.slStyle == "Normal" else refLow2
            sl = rawSL - slAtr

            risk = entry - sl
            if risk <= 0:
                sl = entry - float(atr14.iat[idx])
                risk = entry - sl

            tp1_R = entry + risk * 1.0
            tp2_R = entry + risk * 2.0
            tp3_R = entry + risk * 3.0
            if bool(trendBull.iat[idx]) and ch50 > 0.15:
                tp3_R = entry + risk * 4.0

            candHigh1 = lastHigh if lastHigh > entry else np.nan
            candHigh2 = lastBear if (lastBear is not None and lastBear > entry) else np.nan

            vals = [v for v in [candHigh1, candHigh2] if not (isinstance(v, float) and math.isnan(v))]
            structNear = min(vals) if vals else np.nan
            structFar = max(vals) if vals else np.nan

            if cfg.tpStyle == "R-multiple":
                tp1, tp2, tp3 = tp1_R, tp2_R, tp3_R
            elif cfg.tpStyle == "Structural":
                tp1 = tp1_R if math.isnan(structNear) else float(structNear)
                tp2 = tp2_R if math.isnan(structFar) else float(structFar)
                tp3 = tp3_R
            else:
                tp1 = tp1_R
                tp2 = tp2_R if math.isnan(structNear) else float(structNear)
                tp3 = tp3_R if math.isnan(structFar) else float(structFar)

            deal_status = "BUY активна — угода в роботі"

        elif bearSignal:
            refHigh1 = high_i
            refHigh2 = lastHigh
            rawSLs = refHigh1 if cfg.slStyle == "Aggressive" else max(refHigh1, refHigh2) if cfg.slStyle == "Normal" else refHigh2
            sl = rawSLs + slAtr

            riskS = sl - entry
            if riskS <= 0:
                sl = entry + float(atr14.iat[idx])
                riskS = sl - entry

            tp1_Rs = entry - riskS * 1.0
            tp2_Rs = entry - riskS * 2.0
            tp3_Rs = entry - riskS * 3.0
            if bool(trendBear.iat[idx]) and ch50 < -0.15:
                tp3_Rs = entry - riskS * 4.0

            candLow1 = lastLow if lastLow < entry else np.nan
            candLow2 = lastBull if (lastBull is not None and lastBull < entry) else np.nan

            vals = [v for v in [candLow1, candLow2] if not (isinstance(v, float) and math.isnan(v))]
            structNearS = max(vals) if vals else np.nan
            structFarS = min(vals) if vals else np.nan

            if cfg.tpStyle == "R-multiple":
                tp1, tp2, tp3 = tp1_Rs, tp2_Rs, tp3_Rs
            elif cfg.tpStyle == "Structural":
                tp1 = tp1_Rs if math.isnan(structNearS) else float(structNearS)
                tp2 = tp2_Rs if math.isnan(structFarS) else float(structFarS)
                tp3 = tp3_Rs
            else:
                tp1 = tp1_Rs
                tp2 = tp2_Rs if math.isnan(structNearS) else float(structNearS)
                tp3 = tp3_Rs if math.isnan(structFarS) else float(structFarS)

            deal_status = "SELL активна — угода в роботі"

    extra = {
        "entryStyle": entryStyle,
        "inPD": ("Discount" if inDisc_i else "Premium" if inPrem_i else "Middle"),
        "manip": ("SPRING" if manipBuy else "UTAD" if manipSell else "—"),
        "struct": ("UP" if structUp else "DOWN" if structDown else "RANGE/MIXED"),
    }

    return SignalResult(
        bull=bullSignal,
        bear=bearSignal,
        style=entryStyle,
        entry=entry,
        sl=sl,
        tp1=tp1, tp2=tp2, tp3=tp3,
        deal_status=deal_status,
        extra=extra
    )

# ==============================
# 📥 DATA (Bybit)
# ==============================

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 350) -> pd.DataFrame:
    def _fetch():
        # Bybit swap часто як "BTC/USDT:USDT"
        market_symbol = symbol.replace("USDT", "/USDT:USDT")
        try:
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, limit=limit)
        except Exception:
            market_symbol = symbol.replace("USDT", "/USDT")
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, limit=limit)
        return ohlcv

    ohlcv = await asyncio.to_thread(_fetch)
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

# ==============================
# 🧩 SETTINGS UI (як на фото)
# ==============================

def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)

def build_settings_kb(cfg: SniperConfig) -> InlineKeyboardMarkup:
    on, off = "✅", "⬜️"
    def t(v: bool) -> str: return on if v else off

    rows = [
        [_btn(f"{t(cfg.onlyClose)} Сигнал тільки після закриття", "tog:onlyClose")],
        [_btn(f"{t(cfg.useEMAtrend)} Фільтр EMA 50/200 (тренд)", "tog:useEMAtrend")],
        [_btn(f"{t(cfg.usePD)} Premium / Discount зона", "tog:usePD")],
        [_btn(f"{t(cfg.useLiq)} Забір ліквідності (sweep)", "tog:useLiq")],
        [_btn(f"{t(cfg.useFVG)} FVG для стилю A", "tog:useFVG")],
        [_btn(f"{t(cfg.useATR)} ATR фільтр свічки", "tog:useATR")],
        [_btn(f"pdLen: {cfg.pdLen}", "edit:pdLen"), _btn(f"BOS: {cfg.bosLookback}", "edit:bosLookback")],
        [_btn(f"structLen: {cfg.structLen}", "edit:structLen")],
        [_btn(f"{t(cfg.useEMAconfirm)} EMA 9/21 підтвердження імпульсу", "tog:useEMAconfirm")],
        [_btn(f"{t(cfg.useManip)} Маніпуляції (SPRING/UTAD)", "tog:useManip")],
        [_btn(f"SL style: {cfg.slStyle}", "cycle:slStyle"), _btn(f"TP style: {cfg.tpStyle}", "cycle:tpStyle")],
        [_btn(f"ATR SL mult: {cfg.baseAtrMult:.2f}", "edit:baseAtrMult")],
        [_btn("⬅️ Закрити", "close")]
    ]
    return InlineKeyboardMarkup(rows)

def cfg_toggle(cfg: SniperConfig, field: str) -> None:
    if hasattr(cfg, field) and isinstance(getattr(cfg, field), bool):
        setattr(cfg, field, not getattr(cfg, field))

def cfg_cycle(cfg: SniperConfig, field: str) -> None:
    if field == "slStyle":
        opts = ["Aggressive", "Normal", "Safe"]
        cfg.slStyle = opts[(opts.index(cfg.slStyle) + 1) % len(opts)] if cfg.slStyle in opts else "Normal"
    elif field == "tpStyle":
        opts = ["R-multiple", "Structural", "Combined"]
        cfg.tpStyle = opts[(opts.index(cfg.tpStyle) + 1) % len(opts)] if cfg.tpStyle in opts else "Combined"

def cfg_edit_bump(cfg: SniperConfig, field: str, direction: int) -> None:
    if field == "pdLen":
        cfg.pdLen = int(max(10, min(300, cfg.pdLen + direction * 5)))
    elif field == "bosLookback":
        cfg.bosLookback = int(max(2, min(30, cfg.bosLookback + direction * 1)))
    elif field == "structLen":
        cfg.structLen = int(max(1, min(20, cfg.structLen + direction * 1)))
    elif field == "baseAtrMult":
        cfg.baseAtrMult = float(max(0.1, min(5.0, cfg.baseAtrMult + direction * 0.1)))

def build_edit_kb(field: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [_btn("➖", f"bump:{field}:-1"), _btn("➕", f"bump:{field}:1")],
        [_btn("⬅️ Назад", "back:settings")]
    ])

# ==============================
# 📣 MESSAGE FORMAT
# ==============================

def fmt_price(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if abs(x) >= 1000:
        return f"{x:.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

def build_signal_text(symbol: str, tf: str, res: SignalResult) -> str:
    side = "BUY" if res.bull else "SELL" if res.bear else "—"
    lines = [
        f"📌 *ICT Sniper сигнал*: *{side}*",
        f"• Пара: *{symbol}*  | TF: *{tf}*",
        f"• Стиль входу: *{res.style}*",
        f"• PD: `{res.extra.get('inPD','—')}` | Struct: `{res.extra.get('struct','—')}` | Manip: `{res.extra.get('manip','—')}`",
        "",
        f"💰 Entry: *{fmt_price(res.entry)}*",
    ]
    if res.sl is not None:
        lines.append(f"🛑 SL: *{fmt_price(res.sl)}*")
    if res.tp1 is not None:
        lines.append(f"🎯 TP1: *{fmt_price(res.tp1)}*")
    if res.tp2 is not None:
        lines.append(f"🎯 TP2: *{fmt_price(res.tp2)}*")
    if res.tp3 is not None:
        lines.append(f"🎯 TP3: *{fmt_price(res.tp3)}*")

    lines += ["", f"📍 Статус: *{res.deal_status}*"]
    return "\n".join(lines)

# ==============================
# 🤖 COMMANDS
# ==============================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "Готово ✅\n"
        "Команди:\n"
        "/run — старт сканера\n"
        "/stop — стоп\n"
        "/settings — налаштування як в TradingView\n"
        "/symbols — показати список пар\n"
        "/set_symbols BTCUSDT,ETHUSDT,... — задати пари\n"
        "/tf 15m|1h|4h — таймфрейм\n"
    )

async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: SniperConfig = STATE["cfg"]  # type: ignore
    await update.message.reply_text("⚙️ Налаштування Alpha Engine v4", reply_markup=build_settings_kb(cfg))

async def on_settings_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    cfg: SniperConfig = STATE["cfg"]  # type: ignore
    data = query.data or ""

    if data == "close":
        await query.edit_message_text("Закрито ✅")
        return

    if data.startswith("tog:"):
        field = data.split(":", 1)[1]
        cfg_toggle(cfg, field)
        await query.edit_message_reply_markup(reply_markup=build_settings_kb(cfg))
        return

    if data.startswith("cycle:"):
        field = data.split(":", 1)[1]
        cfg_cycle(cfg, field)
        await query.edit_message_reply_markup(reply_markup=build_settings_kb(cfg))
        return

    if data.startswith("edit:"):
        field = data.split(":", 1)[1]
        await query.edit_message_text(
            f"✏️ Зміна `{field}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_edit_kb(field)
        )
        return

    if data.startswith("bump:"):
        _, field, dir_s = data.split(":")
        cfg_edit_bump(cfg, field, int(dir_s))
        await query.edit_message_text(
            f"✏️ Зміна `{field}` → `{getattr(cfg, field)}`",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_edit_kb(field)
        )
        return

    if data == "back:settings":
        await query.edit_message_text("⚙️ Налаштування Alpha Engine v4", reply_markup=build_settings_kb(cfg))
        return

async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: SniperConfig = STATE["cfg"]  # type: ignore
    await update.message.reply_text("Пари:\n" + ", ".join(cfg.symbols))

async def cmd_set_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: SniperConfig = STATE["cfg"]  # type: ignore
    if not context.args:
        await update.message.reply_text("Приклад: /set_symbols BTCUSDT,ETHUSDT,SOLUSDT")
        return
    raw = " ".join(context.args)
    parts = [p.strip().upper() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        await update.message.reply_text("Не бачу пар. Приклад: /set_symbols BTCUSDT,ETHUSDT")
        return
    cfg.symbols = tuple(parts)
    await update.message.reply_text("✅ Оновлено: " + ", ".join(cfg.symbols))

async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: SniperConfig = STATE["cfg"]  # type: ignore
    if not context.args:
        await update.message.reply_text(f"Поточний TF: {cfg.timeframe}\nПриклад: /tf 15m")
        return
    tf = context.args[0].strip()
    allowed = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}
    if tf not in allowed:
        await update.message.reply_text("Невірний TF. Дозволено: " + ", ".join(sorted(allowed)))
        return
    cfg.timeframe = tf
    await update.message.reply_text("✅ TF = " + tf)

async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if STATE["running"]:
        await update.message.reply_text("Сканер вже працює ✅")
        return

    STATE["running"] = True
    STATE["chat_id"] = update.effective_chat.id
    cfg: SniperConfig = STATE["cfg"]  # type: ignore

    if cfg.universe == "all":
        await update.message.reply_text("🔎 Завантажую всі USDT Perpetual пари з Bybit...")
        STATE["ALL_SYMBOLS"] = await load_all_usdt_perp_symbols()
        await update.message.reply_text(
            f"✅ Завантажено {len(STATE['ALL_SYMBOLS'])} пар"
        )

    task = asyncio.create_task(scanner_loop(context.application))
    STATE["task"] = task

    await update.message.reply_text("🚀 Запустив сканер. Сигнали прийдуть у цей чат.")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    STATE["running"] = False
    task = STATE.get("task")
    if task:
        try:
            task.cancel()
        except Exception:
            pass
    STATE["task"] = None
    await update.message.reply_text("🛑 Зупинив сканер.")

# ==============================
# 🔁 SCANNER LOOP (FIXED)
# ==============================

def _signal_key(symbol: str, tf: str) -> str:
    return f"{symbol}:{tf}"

async def scanner_loop(app: Application) -> None:
    while STATE["running"]:
        cfg: SniperConfig = STATE["cfg"]  # type: ignore
        chat_id = STATE.get("chat_id")

        if not chat_id:
            await asyncio.sleep(cfg.scan_interval_sec)
            continue

        symbols = cfg.symbols
        if cfg.universe == "all" or not symbols:
            symbols = STATE.get("ALL_SYMBOLS", ())

        for symbol in symbols:
            try:
                df = await fetch_ohlcv(symbol, cfg.timeframe, limit=350)
                res = compute_signal(df, cfg)

                if not (res.bull or res.bear):
                    continue

                last_ts = int(df["ts"].iat[-2 if cfg.onlyClose else -1])
                key = _signal_key(symbol, cfg.timeframe)

                side = "BUY" if res.bull else "SELL"
                last = STATE["last_signal"].get(key)

                if last and last.get("ts") == last_ts and last.get("side") == side:
                    continue

                STATE["last_signal"][key] = {"ts": last_ts, "side": side}

                text = build_signal_text(symbol, cfg.timeframe, res)
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[SCAN ERROR] {symbol}: {e}")

        await asyncio.sleep(cfg.scan_interval_sec)
# ==============================
# 🧷 MAIN
# ==============================

def main() -> None:
    application = Application.builder().token(TG_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("run", cmd_run))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CallbackQueryHandler(on_settings_click))
    application.add_handler(CommandHandler("symbols", cmd_symbols))
    application.add_handler(CommandHandler("set_symbols", cmd_set_symbols))
    application.add_handler(CommandHandler("tf", cmd_tf))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()





