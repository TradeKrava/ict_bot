# ==============================
# ICT SNIPER SCANNER — BYBIT USDT PERP
# Alpha Engine v4 logic (A/B/C + SL/TP) — Python port (core)
# python-telegram-bot v21+
# ==============================

import os
import asyncio
import math
import ccxt
import numpy as np
import pandas as pd

from dataclasses import dataclass
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ==============================
# 🔑 TELEGRAM TOKEN
# ==============================
TOKEN = os.getenv("TG_TOKEN")
if not TOKEN:
    raise RuntimeError("Set TG_TOKEN env var")

# ==============================
# 📊 BYBIT (USDT PERP)
# ==============================
exchange = ccxt.bybit({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}  # USDT Perpetual
})

# ==============================
# ⚙️ STATE (як у твоєму стилі)
# ==============================
STATE = {
    "running": False,
    "chat_id": None,

    "tfs": ["15m", "1h"],

    "universe": "all",
    "whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],

    "only_close": True,

    "use_pd": False,
    "use_atr": True,
    "use_fvg": True,
    "use_manip": False,

    "use_ema_trend": True,
    "use_ema_confirm": False,

    "pdLen": 50,
    "bosLookback": 5,
    "structLen": 3,

    "use_sltp": True,
    "slStyle": "Normal",
    "tpStyle": "Combined",
    "usePartial": True,
    "baseAtrMult": 0.6,

    "last_signal": {}
}



# ==============================
# 🧮 Helpers: EMA / ATR / pivots
# ==============================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return true_range(df).ewm(span=length, adjust=False).mean()

def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    # Pine ta.pivothigh(high, L, L): значення з'являється на барі (t), але відноситься до t-L
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = high.iloc[i - left:i + right + 1]
        h = high.iloc[i]
        if h == window.max() and (window == h).sum() == 1:
            out[i] = h
    return pd.Series(out, index=high.index)

def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = low.iloc[i - left:i + right + 1]
        l = low.iloc[i]
        if l == window.min() and (window == l).sum() == 1:
            out[i] = l
    return pd.Series(out, index=low.index)

# ==============================
# 📦 Data fetch
# ==============================
async def fetch_ohlcv(symbol: str, tf: str, limit: int = 300) -> pd.DataFrame:
    # ccxt sync -> в asyncio винесемо в thread, але для простоти тут run_in_executor не робимо.
    data = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

# ==============================
# 🧠 CORE LOGIC: Alpha Engine v4 (signals + SL/TP)
# ==============================
@dataclass
class SignalResult:
    symbol: str
    tf: str
    direction: str          # "BUY" or "SELL"
    entry_style: str        # "A — Conservative" / "B — Normal" / "C — Aggressive"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    info: dict

def compute_last_fvg_levels(df: pd.DataFrame):
    # bullFVG = high[2] < low  -> lastBullFVG := low
    # bearFVG = low[2] > high  -> lastBearFVG := high
    lastBullFVG = np.nan
    lastBearFVG = np.nan
    highs = df["high"].values
    lows = df["low"].values
    for i in range(2, len(df)):
        bullFVG = highs[i-2] < lows[i]
        bearFVG = lows[i-2] > highs[i]
        if bullFVG:
            lastBullFVG = lows[i]
        if bearFVG:
            lastBearFVG = highs[i]
    return lastBullFVG, lastBearFVG

def retest_bull_fvg(level: float, low: float, close: float) -> bool:
    return (not math.isnan(level)) and (low <= level) and (close > level)

def retest_bear_fvg(level: float, high: float, close: float) -> bool:
    return (not math.isnan(level)) and (high >= level) and (close < level)

def sltp_engine(df: pd.DataFrame, idx: int, dir_active: int,
                lastHigh: float, lastLow: float,
                lastBullFVG: float, lastBearFVG: float,
                ema21_val: float,
                slStyle: str, tpStyle: str,
                baseAtrMult: float,
                manip_flag: bool,
                trendBull: bool, trendBear: bool,
                ch50: float) -> tuple[float,float,float,float,float]:
    """
    Порт SL/TP частини з Pine (для нового сигналу).
    idx - індекс останньої закритої свічки.
    """
    close = float(df.loc[idx, "close"])
    high  = float(df.loc[idx, "high"])
    low   = float(df.loc[idx, "low"])

    atr5 = float(df.loc[idx, "atr5"])
    atr14 = float(df.loc[idx, "atr14"])

    # 1) styleMult
    styleMult = 0.5 if slStyle == "Aggressive" else (1.0 if slStyle == "Normal" else 1.5)

    # 2) dynMult (спрощено 1:1 за умовами Pine)
    dynMult = 1.0
    if manip_flag:
        dynMult += 0.4

    dist21 = abs(close - ema21_val) / close
    dist50 = abs(close - float(df.loc[idx, "ema50"])) / close

    if dist21 < 0.004:
        dynMult -= 0.25
    elif dist50 < 0.006:
        dynMult -= 0.15

    dynMult = max(0.5, min(dynMult, 2.0))

    slAtrMult = baseAtrMult * styleMult * dynMult
    slAtr = atr5 * slAtrMult

    entry = close

    if dir_active == 1:
        # BUY
        refLow1 = low
        refLow2 = low if math.isnan(lastLow) else lastLow

        if slStyle == "Aggressive":
            rawSL = refLow1
        elif slStyle == "Normal":
            rawSL = min(refLow1, refLow2)
        else:
            rawSL = refLow2

        sl = rawSL - slAtr

        risk = entry - sl
        if risk <= 0:
            sl = entry - atr14
            risk = entry - sl

        tp1_R = entry + risk * 1.0
        tp2_R = entry + risk * 2.0
        tp3_R = entry + risk * 3.0
        if trendBull and ch50 > 0.15:
            tp3_R = entry + risk * 4.0

        candHigh1 = lastHigh if (not math.isnan(lastHigh) and lastHigh > entry) else np.nan
        candHigh2 = lastBearFVG if (not math.isnan(lastBearFVG) and lastBearFVG > entry) else np.nan

        structNear = np.nan
        structFar  = np.nan
        cands = [x for x in [candHigh1, candHigh2] if not math.isnan(x)]
        if cands:
            structNear = min(cands)
            structFar  = max(cands)

        if tpStyle == "R-multiple":
            tp1, tp2, tp3 = tp1_R, tp2_R, tp3_R
        elif tpStyle == "Structural":
            tp1 = tp1_R if math.isnan(structNear) else structNear
            tp2 = tp2_R if math.isnan(structFar)  else structFar
            tp3 = tp3_R
        else:  # Combined
            tp1 = tp1_R
            tp2 = tp2_R if math.isnan(structNear) else structNear
            tp3 = tp3_R if math.isnan(structFar)  else structFar

        return entry, sl, tp1, tp2, tp3

    else:
        # SELL
        refHigh1 = high
        refHigh2 = high if math.isnan(lastHigh) else lastHigh

        if slStyle == "Aggressive":
            rawSL = refHigh1
        elif slStyle == "Normal":
            rawSL = max(refHigh1, refHigh2)
        else:
            rawSL = refHigh2

        sl = rawSL + slAtr

        risk = sl - entry
        if risk <= 0:
            sl = entry + atr14
            risk = sl - entry

        tp1_R = entry - risk * 1.0
        tp2_R = entry - risk * 2.0
        tp3_R = entry - risk * 3.0
        if trendBear and ch50 < -0.15:
            tp3_R = entry - risk * 4.0

        candLow1 = lastLow if (not math.isnan(lastLow) and lastLow < entry) else np.nan
        candLow2 = lastBullFVG if (not math.isnan(lastBullFVG) and lastBullFVG < entry) else np.nan

        structNear = np.nan
        structFar  = np.nan
        cands = [x for x in [candLow1, candLow2] if not math.isnan(x)]
        if cands:
            structNear = max(cands)
            structFar  = min(cands)

        if tpStyle == "R-multiple":
            tp1, tp2, tp3 = tp1_R, tp2_R, tp3_R
        elif tpStyle == "Structural":
            tp1 = tp1_R if math.isnan(structNear) else structNear
            tp2 = tp2_R if math.isnan(structFar)  else structFar
            tp3 = tp3_R
        else:  # Combined
            tp1 = tp1_R
            tp2 = tp2_R if math.isnan(structNear) else structNear
            tp3 = tp3_R if math.isnan(structFar)  else structFar

        return entry, sl, tp1, tp2, tp3

def analyze_symbol(df: pd.DataFrame, symbol: str, tf: str) -> SignalResult | None:
    # Працюємо по останній ЗАКРИТІЙ свічці
    if len(df) < 220:
        return None

    # якщо only_close: беремо передостанній рядок (бо останній може формуватись)
    idx = df.index[-2] if STATE["only_close"] else df.index[-1]

    # ---------------- PD ----------------
    pdLen = STATE["pdLen"]
    window = df.loc[:idx].tail(pdLen)
    sHigh = float(window["high"].max())
    sLow  = float(window["low"].min())
    midPD = (sHigh + sLow) / 2.0

    close = float(df.loc[idx, "close"])
    high  = float(df.loc[idx, "high"])
    low   = float(df.loc[idx, "low"])

    inDiscount = close < midPD
    inPremium  = close > midPD

    # ---------------- sweeps (liq) ----------------
    prev5 = df.loc[:idx].tail(6).iloc[:-1]  # 5 барів ДО поточного
    liqLow  = low  < float(prev5["low"].min())
    liqHigh = high > float(prev5["high"].max())

    # ---------------- ATR pass ----------------
    atr14 = float(df.loc[idx, "atr14"])
    atrPass = (not STATE["use_atr"]) or ((high - low) >= atr14 * 0.8)

    # ---------------- FVG ----------------
    lastBullFVG, lastBearFVG = compute_last_fvg_levels(df.loc[:idx].copy())
    bullRetestFVG_A = retest_bull_fvg(lastBearFVG, low, close)   # bull ретест верхнього FVG
    bearRetestFVG_A = retest_bear_fvg(lastBullFVG, high, close)  # bear ретест нижнього FVG

    # ---------------- EMA filters ----------------
    ema9   = float(df.loc[idx, "ema9"])
    ema21  = float(df.loc[idx, "ema21"])
    ema50  = float(df.loc[idx, "ema50"])
    ema200 = float(df.loc[idx, "ema200"])

    trendBull = (ema50 > ema200) and (close > ema50)
    trendBear = (ema50 < ema200) and (close < ema50)

    confirmBull = ema9 > ema21
    confirmBear = ema9 < ema21

    finalBull_OK = (not STATE["use_ema_trend"] or trendBull) and (not STATE["use_ema_confirm"] or confirmBull)
    finalBear_OK = (not STATE["use_ema_trend"] or trendBear) and (not STATE["use_ema_confirm"] or confirmBear)

    # ---------------- STRUCT pivots + lastHigh/lastLow ----------------
    # Піводи вже пораховані в df["ph"]/df["pl"]
    sub = df.loc[:idx]
    lastHigh = sub["lastHigh"].iloc[-1]
    lastLow  = sub["lastLow"].iloc[-1]

    # ---------------- BOS ICT ----------------
    bosLookback = STATE["bosLookback"]
    prevN = sub.tail(bosLookback + 1).iloc[:-1]
    bosUpIct   = liqLow  and (close > float(prevN["high"].max()))
    bosDownIct = liqHigh and (close < float(prevN["low"].min()))

    # ---------------- Manip (SPRING/UTAD) ----------------
    # У Pine: bosUpAfterSweep = sweepLow and close > lastHigh
    # Тут: якщо є sweepLow і lastHigh валідний і close вище
    manipBuy = STATE["use_manip"] and liqLow and (not pd.isna(lastHigh)) and (close > float(lastHigh))
    manipSell = STATE["use_manip"] and liqHigh and (not pd.isna(lastLow)) and (close < float(lastLow))
    manip_flag = manipBuy or manipSell

    # ---------------- Entry A/B/C ----------------
    usePD = STATE["use_pd"]
    useFVG = STATE["use_fvg"]

    bullA = bosUpIct   and (bullRetestFVG_A if useFVG else True) and (not usePD or inDiscount) and finalBull_OK and atrPass
    bearA = bosDownIct and (bearRetestFVG_A if useFVG else True) and (not usePD or inPremium)  and finalBear_OK and atrPass

    bullB = bosUpIct   and (not usePD or inDiscount) and finalBull_OK and atrPass
    bearB = bosDownIct and (not usePD or inPremium)  and finalBear_OK and atrPass

    bullC = bosUpIct   and atrPass
    bearC = bosDownIct and atrPass

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

    if not (bullSignal or bearSignal):
        return None

    # ---------------- SL/TP ----------------
    if not STATE["use_sltp"]:
        return None

    # ch50 як у Pine: ta.change(ema50,5)/ema50*100
    ema50_prev5 = float(sub["ema50"].iloc[-6]) if len(sub) >= 6 else ema50
    ch50 = (ema50 - ema50_prev5) / ema50 * 100.0 if ema50 != 0 else 0.0

    direction = "BUY" if bullSignal else "SELL"
    dir_active = 1 if bullSignal else -1

    entry, sl, tp1, tp2, tp3 = sltp_engine(
        df=df, idx=idx, dir_active=dir_active,
        lastHigh=float(lastHigh) if not pd.isna(lastHigh) else np.nan,
        lastLow=float(lastLow) if not pd.isna(lastLow) else np.nan,
        lastBullFVG=float(lastBullFVG) if not math.isnan(lastBullFVG) else np.nan,
        lastBearFVG=float(lastBearFVG) if not math.isnan(lastBearFVG) else np.nan,
        ema21_val=ema21,
        slStyle=STATE["slStyle"],
        tpStyle=STATE["tpStyle"],
        baseAtrMult=float(STATE["baseAtrMult"]),
        manip_flag=manip_flag,
        trendBull=trendBull, trendBear=trendBear,
        ch50=ch50
    )

    info = {
        "trend": "UP" if trendBull else ("DOWN" if trendBear else "RANGE"),
        "pd": "DISCOUNT" if inDiscount else ("PREMIUM" if inPremium else "MID"),
        "liq": "SWEEP_LOW" if liqLow else ("SWEEP_HIGH" if liqHigh else "—"),
        "manip": "SPRING" if manipBuy else ("UTAD" if manipSell else "—"),
    }

    return SignalResult(
        symbol=symbol, tf=tf,
        direction=direction,
        entry_style=entryStyle,
        entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        info=info
    )

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    df["atr14"] = atr(df, 14)
    df["atr5"] = atr(df, 5)

    # pivots
    L = STATE["structLen"]
    df["ph"] = pivot_high(df["high"], L, L)
    df["pl"] = pivot_low(df["low"], L, L)

    # lastHigh/lastLow як var в Pine
    lastHigh = np.nan
    prevHigh = np.nan
    lastLow  = np.nan
    prevLow  = np.nan

    lastHigh_arr = []
    lastLow_arr = []

    for i in range(len(df)):
        if not math.isnan(df["ph"].iloc[i]):
            prevHigh = lastHigh
            lastHigh = float(df["ph"].iloc[i])
        if not math.isnan(df["pl"].iloc[i]):
            prevLow = lastLow
            lastLow = float(df["pl"].iloc[i])
        lastHigh_arr.append(lastHigh)
        lastLow_arr.append(lastLow)

    df["lastHigh"] = lastHigh_arr
    df["lastLow"] = lastLow_arr
    return df

# ==============================
# 🔎 SYMBOL UNIVERSE
# ==============================
def get_symbols() -> list[str]:
    if STATE["universe"] == "whitelist":
        return STATE["whitelist"]

    markets = exchange.load_markets()
    syms = []
    for s, m in markets.items():
        # USDT swap symbols на Bybit в ccxt часто виглядають як "BTC/USDT:USDT"
        if m.get("swap") and m.get("quote") == "USDT":
            syms.append(s)

    # top100: приблизно по об'єму не візьмемо ідеально без дод. API,
    # тому робимо pragmatic: просто перші 100 (або можеш замінити на tickers+sort).
    if STATE["universe"] == "top100":
        return syms[:100]
    return syms

# ==============================
# 📣 Telegram messaging
# ==============================
def format_signal(sig: SignalResult) -> str:
    return (
        f"🚨 *{sig.direction} ICT*  ({sig.entry_style})\n"
        f"• *{sig.symbol}*  TF: *{sig.tf}*\n"
        f"• Entry: `{sig.entry:.6f}`\n"
        f"• SL: `{sig.sl:.6f}`\n"
        f"• TP1: `{sig.tp1:.6f}`\n"
        f"• TP2: `{sig.tp2:.6f}`\n"
        f"• TP3: `{sig.tp3:.6f}`\n"
        f"—\n"
        f"Trend: *{sig.info.get('trend')}* | PD: *{sig.info.get('pd')}* | Liq: *{sig.info.get('liq')}* | Manip: *{sig.info.get('manip')}*"
    )

async def send_signal(app: Application, chat_id: int, sig: SignalResult, bar_ts: pd.Timestamp):
    key = (sig.symbol, sig.tf, sig.direction)
    last = STATE["last_signal"].get(key)

    # антидубль: той же бар — не шлемо
    ts_ms = int(bar_ts.value // 10**6)
    if last == ts_ms:
        return

    STATE["last_signal"][key] = ts_ms
    await app.bot.send_message(chat_id=chat_id, text=format_signal(sig), parse_mode="Markdown")

# ==============================
# 🔁 Scanner loop
# ==============================
async def scanner_loop(app: Application):
    while True:
        if not STATE["running"] or not STATE["chat_id"]:
            await asyncio.sleep(1.0)
            continue

        try:
            symbols = get_symbols()
            for symbol in symbols:
                if not STATE["running"]:
                    break

                for tf in STATE["tfs"]:
                    if not STATE["running"]:
                        break

                    try:
                        df = await fetch_ohlcv(symbol, tf, limit=320)
                        df = prepare_indicators(df)

                        sig = analyze_symbol(df, symbol, tf)
                        if sig:
                            # останній закритий бар
                            idx = df.index[-2] if STATE["only_close"] else df.index[-1]
                            bar_ts = df.loc[idx, "ts"]
                            await send_signal(app, STATE["chat_id"], sig, bar_ts)

                        # маленька пауза щоб не душити rate limit
                        await asyncio.sleep(0.08)

                    except Exception:
                        # щоб не падало через одну пару
                        continue

            # пауза між повними проходами (під твій стиль)
            await asyncio.sleep(3.0)

        except Exception:
            await asyncio.sleep(3.0)

# ==============================
# 🤖 Telegram commands
# ==============================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["chat_id"] = update.effective_chat.id
    await update.message.reply_text(
        "Готово ✅\nКоманди:\n"
        "/startscan — запуск сканера\n"
        "/stopscan — стоп\n"
        "/status — статус\n"
        "/tfs 5m 15m 1h — таймфрейми\n"
        "/universe top100|all|whitelist — всесвіт пар"
    )

async def cmd_startscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["chat_id"] = update.effective_chat.id
    STATE["running"] = True
    await update.message.reply_text("Сканер запущено ✅")

async def cmd_stopscan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATE["running"] = False
    await update.message.reply_text("Сканер зупинено 🛑")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"running={STATE['running']}\n"
        f"tfs={STATE['tfs']}\n"
        f"universe={STATE['universe']}\n"
        f"only_close={STATE['only_close']}\n"
        f"SL={STATE['slStyle']} TP={STATE['tpStyle']} ATRmult={STATE['baseAtrMult']}"
    )

async def cmd_tfs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        STATE["tfs"] = context.args
        await update.message.reply_text(f"TFs встановлено: {STATE['tfs']}")
    else:
        await update.message.reply_text(f"Поточні TFs: {STATE['tfs']}")

async def cmd_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args and context.args[0] in ("top100", "all", "whitelist"):
        STATE["universe"] = context.args[0]
        await update.message.reply_text(f"Universe: {STATE['universe']}")
    else:
        await update.message.reply_text("Використання: /universe top100|all|whitelist")

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("startscan", cmd_startscan))
    app.add_handler(CommandHandler("stopscan", cmd_stopscan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("tfs", cmd_tfs))
    app.add_handler(CommandHandler("universe", cmd_universe))

    # запускаємо scanner loop як background task всередині event loop PTB
    async def post_init(application: Application):
        application.create_task(scanner_loop(application))

    app.post_init = post_init
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
