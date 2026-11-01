import os
import json
import time
import requests
import numpy as np
import pandas as pd

ASSETS = {
    "XAUUSD": {"src": "gold", "dec": 2},
    "XAGUSD": {"src": "yahoo", "code": "SI=F", "dec": 3},
    "USOIL": {"src": "yahoo", "code": "CL=F", "dec": 2},
    "US100": {"src": "yahoo", "code": "NQ=F", "dec": 1},
    "EURUSD": {"src": "yahoo", "code": "EURUSD=X", "dec": 5},
    "GBPUSD": {"src": "yahoo", "code": "GBPUSD=X", "dec": 5},
    "USDJPY": {"src": "yahoo", "code": "JPY=X", "dec": 3},
    "BTCUSDT": {"src": "binance", "code": "BTCUSDT", "dec": 2, "crypto": True},
    "ETHUSDT": {"src": "binance", "code": "ETHUSDT", "dec": 2, "crypto": True},
}

TOKEN = os.environ["TG_TOKEN"]
CHAT = os.environ["TG_CHAT"]
SYMBOL = os.environ.get("SYMBOL", "XAUUSD")
MODE = os.environ.get("MODE", "signal")
TD_KEY = os.environ.get("TD_KEY", "")
CFG = ASSETS[SYMBOL]
DEC = CFG["dec"]
BAR_MS = 15 * 60 * 1000
CRYPTO = CFG.get("crypto", False)
ST_F = 3.5 if CRYPTO else 3.0
SL_M = 2.0 if CRYPTO else 1.5
RSI_OB = 75 if CRYPTO else 70
RSI_OS = 25 if CRYPTO else 30
RR2 = 2.0
WARMUP = 250
SHOW = 15
LOOKBACK = 8
MAX_AGE_MIN = 150
STATE_FILE = "state_" + SYMBOL + ".json"


def fmt(x):
    return format(x, "." + str(DEC) + "f")


def send(text):
    url = "https://api.telegram.org/bot" + TOKEN + "/sendMessage"
    r = requests.post(url, data={"chat_id": CHAT, "text": text}, timeout=20)
    r.raise_for_status()


def finish(df):
    for k in ["o", "h", "l", "c"]:
        df[k] = df[k].astype(float)
    df["ot"] = df["ot"].astype("int64")
    df["ct"] = df["ot"] + BAR_MS - 1
    now = int(time.time() * 1000)
    df = df.drop_duplicates("ot").sort_values("ot")
    return df[df["ct"] < now].reset_index(drop=True)


def from_twelve(key):
    params = {
        "symbol": "XAU/USD",
        "interval": "15min",
        "outputsize": 700,
        "timezone": "UTC",
        "order": "ASC",
        "apikey": key,
    }
    j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30).json()
    if "values" not in j:
        raise RuntimeError(str(j)[:200])
    df = pd.DataFrame(j["values"])
    df["ot"] = pd.to_datetime(df["datetime"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
    return finish(df[["ot", "o", "h", "l", "c"]])


def from_yahoo(code):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + code
    r = requests.get(
        url,
        params={"interval": "15m", "range": "30d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ot": [t * 1000 for t in res["timestamp"]],
        "o": q["open"],
        "h": q["high"],
        "l": q["low"],
        "c": q["close"],
    })
    return finish(df.dropna())


def from_binance(symbol):
    url = "https://data-api.binance.vision/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": "15m", "limit": 1000}, timeout=30)
    r.raise_for_status()
    cols = ["ot", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tq", "i"]
    df = pd.DataFrame(r.json(), columns=cols)
    return finish(df[["ot", "o", "h", "l", "c"]])


def get_klines():
    src = CFG["src"]
    if src == "binance":
        return from_binance(CFG["code"]), "binance"
    if src == "gold":
        if TD_KEY:
            try:
                return from_twelve(TD_KEY), "twelvedata"
            except Exception as e:
                print("twelvedata failed:", e)
        return from_yahoo("GC=F"), "yahoo GC=F"
    return from_yahoo(CFG["code"]), "yahoo " + CFG["code"]


def rma(arr, n):
    return pd.Series(arr).ewm(alpha=1.0 / n, adjust=False).mean().values


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def supertrend_dir(h, l, c, factor, n):
    atr = rma(true_range(h, l, c), n)
    hl2 = (h + l) / 2.0
    lo = hl2 - factor * atr
    hi = hl2 + factor * atr
    d = np.ones(len(c))
    for i in range(1, len(c)):
        if not (lo[i] > lo[i - 1] or c[i - 1] < lo[i - 1]):
            lo[i] = lo[i - 1]
        if not (hi[i] < hi[i - 1] or c[i - 1] > hi[i - 1]):
            hi[i] = hi[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] > hi[i] else 1
        else:
            d[i] = 1 if c[i] < lo[i] else -1
    return d


def rsi_calc(c, n):
    diff = np.diff(c, prepend=c[0])
    up = rma(np.where(diff > 0, diff, 0.0), n)
    dn = rma(np.where(diff < 0, -diff, 0.0), n)
    rs = up / np.where(dn == 0, 1e-12, dn)
    return 100 - 100 / (1 + rs)


def main():
    df, src = get_klines()
    h = df["h"].values
    l = df["l"].values
    c = df["c"].values
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().values
    rsi = rsi_calc(c, 14)
    atr = rma(true_range(h, l, c), 14)
    d = supertrend_dir(h, l, c, ST_F, 10)
    times = pd.to_datetime(df["ot"], unit="ms") + pd.Timedelta(hours=7)

    if MODE == "test":
        last = times.iloc[-1].strftime("%d/%m %H:%M")
        send("ทดสอบบอท OK " + SYMBOL + "\nแหล่งข้อมูล " + src
             + "\nแท่งล่าสุด (เวลาไทย) " + last
             + "\nราคาปิด " + fmt(c[-1]))
        return

    def check(i):
        buy = d[i] == -1 and d[i - 1] == 1 and c[i] > ema[i] and 50 < rsi[i] < RSI_OB
        sell = d[i] == 1 and d[i - 1] == -1 and c[i] < ema[i] and RSI_OS < rsi[i] < 50
        if not buy and not sell:
            return None
        sign = 1 if buy else -1
        risk = atr[i] * SL_M
        side = "BUY" if buy else "SELL"
        return (side, c[i], c[i] - sign * risk, c[i] + sign * risk, c[i] + sign * risk * RR2)

    if MODE == "compare":
        rows = []
        for i in range(WARMUP, len(c)):
            s = check(i)
            if s:
                rows.append(s[0] + " " + times.iloc[i].strftime("%d/%m %H:%M")
                            + " E " + fmt(s[1])
                            + " SL " + fmt(s[2])
                            + " TP1 " + fmt(s[3])
                            + " TP2 " + fmt(s[4]))
        head = "เทียบสัญญาณ " + SYMBOL + " M15 (เวลาไทย) แหล่งข้อมูล " + src + "\n"
        if rows:
            send(head + "\n".join(rows[-SHOW:]))
        else:
            send(head + "ไม่มีสัญญาณในช่วงข้อมูลนี้")
        return

    last_sent = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                last_sent = int(json.load(f)["last"])
        except Exception:
            last_sent = 0

    now_ms = time.time() * 1000
    changed = False
    start = max(WARMUP, len(c) - LOOKBACK)
    for i in range(start, len(c)):
        ot = int(df["ot"].iloc[i])
        age = (now_ms - int(df["ct"].iloc[i])) / 60000.0
        if ot <= last_sent or age > MAX_AGE_MIN:
            continue
        s = check(i)
        if not s:
            continue
        late = ""
        if age > 16:
            late = "\n(ล่าช้า " + str(int(age)) + " นาที ราคาล่าสุด " + fmt(c[-1]) + ")"
        send(s[0] + " " + SYMBOL + " M15\n"
             + "แท่ง " + times.iloc[i].strftime("%d/%m %H:%M") + "\n"
             + "Entry " + fmt(s[1]) + "\n"
             + "SL " + fmt(s[2]) + "\n"
             + "TP1 " + fmt(s[3]) + "\n"
             + "TP2 " + fmt(s[4]) + late)
        last_sent = ot
        changed = True

    if changed:
        with open(STATE_FILE, "w") as f:
            json.dump({"last": last_sent}, f)


main()
