import os
import json
import math
import time
import requests
import numpy as np
import pandas as pd

TOKEN = os.environ.get("TG_TOKEN", "")
CHAT = os.environ.get("TG_CHAT", "")
SYMBOL = os.environ.get("SYMBOL", "XAUUSD")
RUN_MODE = os.environ.get("MODE", "signal")
TD_KEY = os.environ.get("TD_KEY", "")
BAR_MS = 15 * 60 * 1000
LOOKBACK = 8
MAX_AGE_MIN = 150
SHOW = 12
STATE_FILE = "state_" + SYMBOL + ".json"
NAN = float("nan")

CFG = {
    "mode": "Normal",
    "level": "Normal",
    "use_fvg": True, "use_brk": True, "use_pb": True, "use_pa": True, "use_rev": True, "use_choch": True,
    "brk_len_n": 20, "brk_len_s": 12, "brk_range": 1.0, "max_ext": 3.5,
    "rev_len": 15, "rev_wick": 0.4, "rev_range": 0.8,
    "pivot_len": 5, "sr_tol": 0.8, "need_sr": True, "min_gap": 0.15, "atr_len": 14,
    "rr_n": 2.0, "sl_buf_n": 0.3, "max_risk_n": 3.0, "cool_n": 8, "max_age_n": 80, "max_hold_n": 40,
    "rr_s": 1.2, "sl_buf_s": 0.15, "max_risk_s": 1.5, "cool_s": 3, "max_age_s": 30, "max_hold_s": 12,
    "use_trend": True, "use_struct": True, "use_imp": True, "imp_len": 5, "imp_atr": 1.0,
    "use_opp": True, "use_htf": True, "use_adx": True, "adx_min": 18.0, "use_rsi": True,
    "use_session": False, "sess_start": 7, "sess_end": 20, "body_min": 0.4, "one_trade": True,
    "use_exit": True, "ex_pattern": True, "ex_strong": True, "ex_ema": True, "ex_rsi": True,
    "strong_atr": 1.0, "ex_min_profit": 0.0,
}


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
    params = {"symbol": "XAU/USD", "interval": "15min", "outputsize": 3000,
              "timezone": "UTC", "order": "ASC", "apikey": key}
    j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=40).json()
    if "values" not in j:
        raise RuntimeError(str(j)[:200])
    df = pd.DataFrame(j["values"])
    df["ot"] = pd.to_datetime(df["datetime"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
    return finish(df[["ot", "o", "h", "l", "c"]])


def from_yahoo(code="GC=F"):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + code
    r = requests.get(url, params={"interval": "15m", "range": "60d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=40)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"ot": [t * 1000 for t in res["timestamp"]],
                       "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]})
    return finish(df.dropna())


def get_klines():
    if TD_KEY:
        try:
            return from_twelve(TD_KEY), "twelvedata"
        except Exception as e:
            print("twelvedata failed:", e)
    return from_yahoo("GC=F"), "yahoo GC=F"


def isnan(x):
    return x is None or x != x


def nz(x, d=0.0):
    return d if isnan(x) else x


def rma(a, n):
    return pd.Series(a).ewm(alpha=1.0 / n, adjust=False).mean().values


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().values


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def rsi_calc(c, n):
    diff = np.diff(c, prepend=c[0])
    up = rma(np.where(diff > 0, diff, 0.0), n)
    dn = rma(np.where(diff < 0, -diff, 0.0), n)
    rs = up / np.where(dn == 0, 1e-12, dn)
    return 100 - 100 / (1 + rs)


def dmi_adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    trur = rma(true_range(h, l, c), n)
    trur = np.where(trur == 0, 1e-12, trur)
    plus = 100 * rma(plus_dm, n) / trur
    minus = 100 * rma(minus_dm, n) / trur
    s = plus + minus
    s = np.where(s == 0, 1.0, s)
    return 100 * rma(np.abs(plus - minus) / s, n)


def rmax(a, n):
    return pd.Series(a).rolling(n, min_periods=n).max().values


def rmin(a, n):
    return pd.Series(a).rolling(n, min_periods=n).min().values


def shift1(a):
    out = np.empty_like(a)
    out[0] = NAN
    out[1:] = a[:-1]
    return out


def htf_ema_series(df):
    t = pd.to_datetime(df["ot"], unit="ms", utc=True)
    hour = pd.DatetimeIndex(t.dt.floor("60min"))
    s = pd.Series(df["c"].values, index=hour)
    hc = s.groupby(level=0).last()
    he = hc.ewm(span=50, adjust=False).mean().shift(1)
    return he.reindex(hour).values


def run_engine(df, cfg=CFG):
    o = df["o"].values.astype(float)
    h = df["h"].values.astype(float)
    l = df["l"].values.astype(float)
    c = df["c"].values.astype(float)
    ot = df["ot"].values
    n = len(c)
    scalp = cfg["mode"] == "Scalping"
    strict = cfg["level"] == "Strict"
    loose = cfg["level"] == "Loose"
    rr = cfg["rr_s"] if scalp else cfg["rr_n"]
    sl_buf = cfg["sl_buf_s"] if scalp else cfg["sl_buf_n"]
    max_risk = cfg["max_risk_s"] if scalp else cfg["max_risk_n"]
    cooldown = cfg["cool_s"] if scalp else cfg["cool_n"]
    max_age = cfg["max_age_s"] if scalp else cfg["max_age_n"]
    max_hold = cfg["max_hold_s"] if scalp else cfg["max_hold_n"]
    brk_len = cfg["brk_len_s"] if scalp else cfg["brk_len_n"]
    use_struct_e = cfg["use_struct"] and strict
    use_htf_e = cfg["use_htf"] and strict
    use_imp_e = cfg["use_imp"] and not loose
    use_opp_e = cfg["use_opp"] and not loose
    use_adx_e = cfg["use_adx"] and not loose
    use_rsi = cfg["use_rsi"]
    need_sr = cfg["need_sr"]

    atr = rma(true_range(h, l, c), cfg["atr_len"])
    body = np.abs(c - o)
    rng = h - l
    low_wick = np.minimum(o, c) - l
    up_wick = h - np.maximum(o, c)
    ef = ema(c, 9 if scalp else 21)
    es = ema(c, 21 if scalp else 50)
    htf = htf_ema_series(df) if use_htf_e else np.full(n, NAN)
    adx = dmi_adx(h, l, c, 14)
    rsi = rsi_calc(c, 14)
    hh = shift1(rmax(h, brk_len))
    ll = shift1(rmin(l, brk_len))
    low_n = shift1(rmin(l, cfg["rev_len"]))
    high_n = shift1(rmax(h, cfg["rev_len"]))
    lowest_cl = rmin(c, 10)
    highest_cl = rmax(c, 10)
    bear_imp = ((c < o) & ((o - c) >= atr * cfg["imp_atr"])).astype(float)
    bull_imp = ((c > o) & ((c - o) >= atr * cfg["imp_atr"])).astype(float)
    bear_imp_n = pd.Series(bear_imp).rolling(cfg["imp_len"], min_periods=1).sum().values
    bull_imp_n = pd.Series(bull_imp).rolling(cfg["imp_len"], min_periods=1).sum().values

    L = cfg["pivot_len"]
    ph = np.full(n, NAN)
    pl = np.full(n, NAN)
    for i in range(2 * L, n):
        ctr = i - L
        if h[ctr] > np.max(h[ctr - L:ctr]) and h[ctr] >= np.max(h[ctr + 1:i + 1]):
            ph[i] = h[ctr]
        if l[ctr] < np.min(l[ctr - L:ctr]) and l[ctr] <= np.min(l[ctr + 1:i + 1]):
            pl[i] = l[ctr]

    times = pd.to_datetime(ot, unit="ms", utc=True)

    def in_sess(i):
        if not cfg["use_session"]:
            return True
        t = times[i]
        return t.weekday() <= 4 and cfg["sess_start"] <= t.hour < cfg["sess_end"]

    R = [NAN] * 4
    S = [NAN] * 4

    def inzone(lv, zb, zt, t):
        return (not isnan(lv)) and lv >= zb - t and lv <= zt + t

    def near_sup(zb, zt, t):
        return any(inzone(v, zb, zt, t) for v in S)

    def near_res(zb, zt, t):
        return any(inzone(v, zb, zt, t) for v in R)

    bT = [NAN] * 3
    bB = [NAN] * 3
    bA = [NAN] * 3
    eT = [NAN] * 3
    eB = [NAN] * 3
    eA = [NAN] * 3

    last_side = "-"
    last_type = ""
    last_entry = NAN
    last_sl = NAN
    last_tp = NAN
    status = "WAIT"
    last_sig_bar = -1000
    n_win = n_loss = n_timeout = n_exit = 0
    net_pts = 0.0
    prev_r1 = NAN
    prev_s1 = NAN
    events = []
    snap = {}

    for i in range(3, n):
        if not isnan(ph[i]):
            R[3], R[2], R[1], R[0] = R[2], R[1], R[0], ph[i]
        if not isnan(pl[i]):
            S[3], S[2], S[1], S[0] = S[2], S[1], S[0], pl[i]
        r1 = R[0]
        s1 = S[0]
        r2 = R[1]
        s2 = S[1]
        a = atr[i]
        tol = a * cfg["sr_tol"]
        op, hi, lo, cl = o[i], h[i], l[i], c[i]
        bd, rg = body[i], rng[i]
        lw, uw = low_wick[i], up_wick[i]

        bull_engulf = c[i - 1] < o[i - 1] and cl > op and cl >= o[i - 1] and op <= c[i - 1] and bd > body[i - 1] and bd >= a * 0.4
        bear_engulf = c[i - 1] > o[i - 1] and cl < op and cl <= o[i - 1] and op >= c[i - 1] and bd > body[i - 1] and bd >= a * 0.4
        pin_bull = rg >= a * 0.5 and lw >= rg * 0.6 and bd <= rg * 0.35 and uw <= rg * 0.25
        pin_bear = rg >= a * 0.5 and uw >= rg * 0.6 and bd <= rg * 0.35 and lw <= rg * 0.25
        morning = c[i - 2] < o[i - 2] and (o[i - 2] - c[i - 2]) >= a * 0.5 and body[i - 1] <= (h[i - 1] - l[i - 1]) * 0.35 and cl > op and cl > (o[i - 2] + c[i - 2]) / 2
        evening = c[i - 2] > o[i - 2] and (c[i - 2] - o[i - 2]) >= a * 0.5 and body[i - 1] <= (h[i - 1] - l[i - 1]) * 0.35 and cl < op and cl < (o[i - 2] + c[i - 2]) / 2
        bull_pat = bull_engulf or pin_bull or morning
        bear_pat = bear_engulf or pin_bear or evening
        pat_bull = "Bull Engulf" if bull_engulf else "Hammer" if pin_bull else "Morning Star" if morning else ""
        pat_bear = "Bear Engulf" if bear_engulf else "Shooting Star" if pin_bear else "Evening Star" if evening else ""

        sess_ok = in_sess(i)
        adx_ok = (not use_adx_e) or adx[i] >= (cfg["adx_min"] - 3 if scalp else cfg["adx_min"])
        good_body = rg > 0 and bd >= rg * cfg["body_min"]
        slope_up = ef[i] > ef[i - 3]
        slope_dn = ef[i] < ef[i - 3]
        if not cfg["use_trend"]:
            trend_buy = trend_sell = True
        elif loose:
            trend_buy = cl > es[i]
            trend_sell = cl < es[i]
        else:
            trend_buy = cl > ef[i] and ef[i] > es[i] and slope_up
            trend_sell = cl < ef[i] and ef[i] < es[i] and slope_dn
        htf_buy = (not use_htf_e) or scalp or cl > htf[i]
        htf_sell = (not use_htf_e) or scalp or cl < htf[i]
        rsi_buy = (not use_rsi) or (rsi[i] > 40 and rsi[i] < 70)
        rsi_sell = (not use_rsi) or (rsi[i] < 60 and rsi[i] > 30)
        no_bear_imp = (not use_imp_e) or bear_imp_n[i] == 0
        no_bull_imp = (not use_imp_e) or bull_imp_n[i] == 0
        cross_up_r1 = cl > r1 and c[i - 1] <= prev_r1
        cross_dn_s1 = cl < s1 and c[i - 1] >= prev_s1
        struct_bull = (not use_struct_e) or ((not isnan(s2)) and s1 >= s2)
        struct_bear = (not use_struct_e) or ((not isnan(r2)) and r1 <= r2)

        sl_hit = status == "ACTIVE" and ((last_side == "BUY" and lo <= last_sl) or (last_side == "SELL" and hi >= last_sl))
        tp_hit = status == "ACTIVE" and ((last_side == "BUY" and hi >= last_tp) or (last_side == "SELL" and lo <= last_tp))
        timeout = status == "ACTIVE" and (not sl_hit) and (not tp_hit) and (i - last_sig_bar) > max_hold
        if last_side == "BUY":
            pts_close = cl - last_entry
        elif last_side == "SELL":
            pts_close = last_entry - cl
        else:
            pts_close = NAN
        profit_ok = pts_close >= a * cfg["ex_min_profit"]
        open_trade = status == "ACTIVE" and (not sl_hit) and (not tp_hit) and (not timeout) and i > last_sig_bar
        ef_prev = ef[i - 1]
        b_buy_pat = profit_ok and cfg["ex_pattern"] and bear_pat
        b_buy_ema = profit_ok and cfg["ex_ema"] and cl < ef[i] and c[i - 1] >= ef_prev
        b_buy_rsi = profit_ok and cfg["ex_rsi"] and rsi[i] < 70 and rsi[i - 1] >= 70
        b_buy_strong = cfg["ex_strong"] and cl < op and (op - cl) >= a * cfg["strong_atr"]
        b_sell_pat = profit_ok and cfg["ex_pattern"] and bull_pat
        b_sell_ema = profit_ok and cfg["ex_ema"] and cl > ef[i] and c[i - 1] <= ef_prev
        b_sell_rsi = profit_ok and cfg["ex_rsi"] and rsi[i] > 30 and rsi[i - 1] <= 30
        b_sell_strong = cfg["ex_strong"] and cl > op and (cl - op) >= a * cfg["strong_atr"]
        exit_buy = cfg["use_exit"] and open_trade and last_side == "BUY" and (b_buy_pat or b_buy_ema or b_buy_rsi or b_buy_strong)
        exit_sell = cfg["use_exit"] and open_trade and last_side == "SELL" and (b_sell_pat or b_sell_ema or b_sell_rsi or b_sell_strong)
        exit_now = exit_buy or exit_sell
        if last_side == "BUY":
            exit_reason = pat_bear if b_buy_pat else "Strong Bear Candle" if b_buy_strong else "Close < EMA" if b_buy_ema else "RSI Reversal"
        else:
            exit_reason = pat_bull if b_sell_pat else "Strong Bull Candle" if b_sell_strong else "Close > EMA" if b_sell_ema else "RSI Reversal"

        active_now = status == "ACTIVE" and (not sl_hit) and (not tp_hit) and (not timeout) and (not exit_now)
        can_trade = (i - last_sig_bar) >= cooldown and ((not cfg["one_trade"]) or (not active_now))

        opp_buy = False
        opp_sell = False
        for k in range(3):
            if (not isnan(eT[k])) and cl >= eB[k] - a * 0.8 and cl <= eT[k]:
                opp_buy = True
            if (not isnan(bT[k])) and cl <= bT[k] + a * 0.8 and cl >= bB[k]:
                opp_sell = True
        no_opp_buy = (not use_opp_e) or (not opp_buy)
        no_opp_sell = (not use_opp_e) or (not opp_sell)

        ok_buy_core = trend_buy and struct_bull and htf_buy and no_bear_imp and no_opp_buy
        ok_sell_core = trend_sell and struct_bear and htf_sell and no_bull_imp and no_opp_sell
        ok_buy_lite = cl > ef[i] and cl > es[i]
        ok_sell_lite = cl < ef[i] and cl < es[i]
        filters_buy = ok_buy_core and adx_ok and rsi_buy and sess_ok and good_body
        filters_sell = ok_sell_core and adx_ok and rsi_sell and sess_ok and good_body
        buy_pa_c = cl > op and (cl > (hi + lo) / 2 or cl > h[i - 1])
        sell_pa_c = cl < op and (cl < (hi + lo) / 2 or cl < l[i - 1])

        hit_b = [False] * 3
        hit_e = [False] * 3
        for k in range(3):
            hit_b[k] = (not isnan(bT[k])) and lo <= bT[k] and cl > bB[k] and buy_pa_c and ((not need_sr) or near_sup(bB[k], bT[k], tol))
            hit_e[k] = (not isnan(eT[k])) and hi >= eB[k] and cl < eT[k] and sell_pa_c and ((not need_sr) or near_res(eB[k], eT[k], tol))
        raw_buy = any(hit_b)
        raw_sell = any(hit_e)
        zone_bot_buy = NAN
        for k in range(3):
            if hit_b[k]:
                zone_bot_buy = bB[k]
                break
        zone_top_sell = NAN
        for k in range(3):
            if hit_e[k]:
                zone_top_sell = eT[k]
                break
        sl_buy = min(nz(zone_bot_buy, lo), lo) - a * sl_buf
        sl_sell = max(nz(zone_top_sell, hi), hi) + a * sl_buf
        risk_ok_buy = (cl - sl_buy) <= a * max_risk and (cl - sl_buy) >= a * 0.25
        risk_ok_sell = (sl_sell - cl) <= a * max_risk and (sl_sell - cl) >= a * 0.25
        buy_fvg = cfg["use_fvg"] and raw_buy and filters_buy and risk_ok_buy and can_trade
        sell_fvg = cfg["use_fvg"] and raw_sell and filters_sell and risk_ok_sell and can_trade

        brk_buy = cfg["use_brk"] and can_trade and (not buy_fvg) and ok_buy_lite and sess_ok and cl > hh[i] and cl > op and rg >= a * cfg["brk_range"] and good_body and (cl - ef[i]) <= a * cfg["max_ext"] and ((not use_rsi) or rsi[i] < 85)
        brk_sell = cfg["use_brk"] and can_trade and (not sell_fvg) and ok_sell_lite and sess_ok and cl < ll[i] and cl < op and rg >= a * cfg["brk_range"] and good_body and (ef[i] - cl) <= a * cfg["max_ext"] and ((not use_rsi) or rsi[i] > 15)
        pb_buy = cfg["use_pb"] and can_trade and (not buy_fvg) and (not brk_buy) and filters_buy and lo <= ef[i] and cl > ef[i] and buy_pa_c
        pb_sell = cfg["use_pb"] and can_trade and (not sell_fvg) and (not brk_sell) and filters_sell and hi >= ef[i] and cl < ef[i] and sell_pa_c
        pa_lvl_buy = near_sup(lo, hi, tol) or (lo <= es[i] and cl > es[i]) or (lo <= ef[i] and cl > ef[i])
        pa_lvl_sell = near_res(lo, hi, tol) or (hi >= es[i] and cl < es[i]) or (hi >= ef[i] and cl < ef[i])
        pa_buy = cfg["use_pa"] and can_trade and (not buy_fvg) and (not brk_buy) and (not pb_buy) and bull_pat and pa_lvl_buy and ok_buy_lite and rsi_buy and sess_ok and no_bear_imp and no_opp_buy and adx_ok and ((not strict) or ok_buy_core)
        pa_sell = cfg["use_pa"] and can_trade and (not sell_fvg) and (not brk_sell) and (not pb_sell) and bear_pat and pa_lvl_sell and ok_sell_lite and rsi_sell and sess_ok and no_bull_imp and no_opp_sell and adx_ok and ((not strict) or ok_sell_core)
        rev_buy_raw = lo < low_n[i] and cl > low_n[i] and cl > op and rg >= a * cfg["rev_range"] and lw >= rg * cfg["rev_wick"] and cl > (hi + lo) / 2
        rev_sell_raw = hi > high_n[i] and cl < high_n[i] and cl < op and rg >= a * cfg["rev_range"] and uw >= rg * cfg["rev_wick"] and cl < (hi + lo) / 2
        rev_buy = cfg["use_rev"] and can_trade and (not buy_fvg) and (not brk_buy) and (not pb_buy) and (not pa_buy) and rev_buy_raw and sess_ok and ((not use_rsi) or rsi[i] < 60)
        rev_sell = cfg["use_rev"] and can_trade and (not sell_fvg) and (not brk_sell) and (not pb_sell) and (not pa_sell) and rev_sell_raw and sess_ok and ((not use_rsi) or rsi[i] > 40)
        ch_buy = cfg["use_choch"] and can_trade and (not buy_fvg) and (not brk_buy) and (not pb_buy) and (not pa_buy) and (not rev_buy) and cross_up_r1 and cl > op and rg >= a * cfg["rev_range"] and good_body and lowest_cl[i] < es[i] and sess_ok and ((not use_rsi) or rsi[i] < 75)
        ch_sell = cfg["use_choch"] and can_trade and (not sell_fvg) and (not brk_sell) and (not pb_sell) and (not pa_sell) and (not rev_sell) and cross_dn_s1 and cl < op and rg >= a * cfg["rev_range"] and good_body and highest_cl[i] > es[i] and sess_ok and ((not use_rsi) or rsi[i] > 25)

        buy_sig = buy_fvg or brk_buy or pb_buy or pa_buy or rev_buy or ch_buy
        sell_fvg_f = sell_fvg and (not buy_sig)
        sell_sig = (sell_fvg or brk_sell or pb_sell or pa_sell or rev_sell or ch_sell) and (not buy_sig)
        is_fvg = (buy_sig and buy_fvg) or (sell_sig and sell_fvg)
        is_brk = (buy_sig and brk_buy) or (sell_sig and brk_sell)
        is_pb = (buy_sig and pb_buy) or (sell_sig and pb_sell)
        is_pa = (buy_sig and pa_buy) or (sell_sig and pa_sell)
        is_rev = (buy_sig and rev_buy) or (sell_sig and rev_sell)
        sig_type = "FVG" if is_fvg else "BRK" if is_brk else "PB" if is_pb else "PA" if is_pa else "REV" if is_rev else "CHoCH"

        sl_buy_alt = min(max(min(lo, l[i - 1]) - a * sl_buf, cl - a * max_risk), cl - a * 0.5)
        sl_sell_alt = max(min(max(hi, h[i - 1]) + a * sl_buf, cl + a * max_risk), cl + a * 0.5)
        sl_now = NAN
        tp_now = NAN
        if buy_sig:
            sl_now = sl_buy if buy_fvg else sl_buy_alt
            tp_now = cl + (cl - sl_now) * rr
        elif sell_sig:
            sl_now = sl_sell if sell_fvg else sl_sell_alt
            tp_now = cl - (sl_now - cl) * rr

        if sl_hit:
            events.append({"i": i, "ot": int(ot[i]), "kind": "SL", "side": last_side, "type": last_type, "entry": last_entry, "sl": last_sl, "tp": last_tp})
        elif tp_hit:
            events.append({"i": i, "ot": int(ot[i]), "kind": "TP", "side": last_side, "type": last_type, "entry": last_entry, "sl": last_sl, "tp": last_tp})
        if timeout:
            events.append({"i": i, "ot": int(ot[i]), "kind": "TIMEOUT", "side": last_side, "type": last_type, "entry": last_entry, "sl": last_sl, "tp": last_tp, "pts": pts_close})
        if exit_now:
            events.append({"i": i, "ot": int(ot[i]), "kind": "EXIT", "side": last_side, "type": last_type, "entry": last_entry, "sl": last_sl, "tp": last_tp, "pts": pts_close, "reason": exit_reason})
        if buy_sig or sell_sig:
            pat = (pat_bull if buy_sig else pat_bear) if sig_type in ("PA", "FVG", "REV", "PB", "BRK", "CHoCH") else ""
            events.append({"i": i, "ot": int(ot[i]), "kind": "BUY" if buy_sig else "SELL", "type": sig_type, "entry": cl, "sl": sl_now, "tp": tp_now, "pat": pat})

        for k in range(3):
            kb = (buy_fvg and hit_b[k]) or ((not isnan(bB[k])) and (cl < bB[k] or i - bA[k] > max_age))
            ke = (sell_fvg_f and hit_e[k]) or ((not isnan(eT[k])) and (cl > eT[k] or i - eA[k] > max_age))
            if kb:
                bT[k] = NAN
                bB[k] = NAN
            if ke:
                eT[k] = NAN
                eB[k] = NAN
        bull_fvg = lo > h[i - 2] and (lo - h[i - 2]) > a * cfg["min_gap"] and c[i - 1] > o[i - 1]
        bear_fvg = hi < l[i - 2] and (l[i - 2] - hi) > a * cfg["min_gap"] and c[i - 1] < o[i - 1]
        if bull_fvg:
            bT[2], bB[2], bA[2] = bT[1], bB[1], bA[1]
            bT[1], bB[1], bA[1] = bT[0], bB[0], bA[0]
            bT[0], bB[0], bA[0] = lo, h[i - 2], float(i)
        if bear_fvg:
            eT[2], eB[2], eA[2] = eT[1], eB[1], eA[1]
            eT[1], eB[1], eA[1] = eT[0], eB[0], eA[0]
            eT[0], eB[0], eA[0] = l[i - 2], hi, float(i)

        risk_pts = abs(last_entry - last_sl)
        reward_pts = abs(last_tp - last_entry)
        if sl_hit:
            trade_pts = -risk_pts
        elif tp_hit:
            trade_pts = reward_pts
        elif exit_now or timeout:
            trade_pts = pts_close
        else:
            trade_pts = 0.0
        net_pts += nz(trade_pts)
        n_win += 1 if ((tp_hit and not sl_hit) or (exit_now and pts_close > 0)) else 0
        n_loss += 1 if (sl_hit or (exit_now and pts_close <= 0)) else 0
        n_timeout += 1 if timeout else 0
        n_exit += 1 if exit_now else 0
        if buy_sig or sell_sig:
            status = "ACTIVE"
        elif sl_hit:
            status = "SL HIT"
        elif tp_hit:
            status = "TP HIT"
        elif timeout:
            status = "TIMEOUT"
        elif exit_now:
            status = "EXIT"
        if buy_sig:
            last_side = "BUY"
        elif sell_sig:
            last_side = "SELL"
        if buy_sig or sell_sig:
            last_type = sig_type
            last_entry = cl
            last_sl = sl_now
            last_tp = tp_now
            last_sig_bar = i
        prev_r1 = r1
        prev_s1 = s1
        snap = {"atr": a, "rsi": rsi[i], "adx": adx[i], "ef": ef[i], "es": es[i], "close": cl,
                "bias": "BULL" if ok_buy_core else "BEAR" if ok_sell_core else "NEUTRAL",
                "s1": s1, "r1": r1}

    st = {"status": status, "side": last_side, "type": last_type, "entry": last_entry, "sl": last_sl, "tp": last_tp,
          "win": n_win, "loss": n_loss, "timeout": n_timeout, "exit": n_exit, "net": net_pts, "snap": snap}
    return events, st


def th(ms):
    return (pd.to_datetime(ms, unit="ms") + pd.Timedelta(hours=7)).strftime("%d/%m %H:%M")


def f2(x):
    return "-" if isnan(x) else format(x, ".2f")


def describe(e):
    k = e["kind"]
    if k in ("BUY", "SELL"):
        risk = abs(e["entry"] - e["sl"])
        rew = abs(e["tp"] - e["entry"])
        rr = rew / risk if risk > 0 else 0
        txt = k + " " + e["type"] + " XAUUSD M15\nแท่ง " + th(e["ot"]) + " (เวลาไทย)\nEntry " + f2(e["entry"]) + "\nSL " + f2(e["sl"]) + " (เสี่ยง " + f2(risk) + ")\nTP " + f2(e["tp"]) + " (RR 1:" + format(rr, ".1f") + ")"
        if e.get("pat"):
            txt += "\nรูปแบบ " + e["pat"]
        return txt
    head = {"SL": "SL HIT", "TP": "TP HIT", "TIMEOUT": "TIMEOUT", "EXIT": "EXIT"}[k]
    txt = head + " " + e["side"] + " " + e["type"] + " XAUUSD M15\nแท่ง " + th(e["ot"]) + " (เวลาไทย)\nEntry " + f2(e["entry"]) + "\nSL " + f2(e["sl"]) + "  TP " + f2(e["tp"])
    if "pts" in e:
        txt += "\nกำไร/ขาดทุนตอนนี้ " + format(e["pts"], "+.2f") + " จุด"
    if e.get("reason"):
        txt += "\nเหตุผล " + e["reason"] + " (ปิดไม้เอง)"
    return txt


def short_line(e):
    k = e["kind"]
    if k in ("BUY", "SELL"):
        return th(e["ot"]) + " " + k + " " + e["type"] + " E " + f2(e["entry"]) + " SL " + f2(e["sl"]) + " TP " + f2(e["tp"])
    return th(e["ot"]) + " " + k + " " + e["side"] + " " + e["type"]


def load_last():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return int(json.load(f)["last"])
        except Exception:
            return 0
    return 0


def main():
    df, src = get_klines()
    events, st = run_engine(df)
    now_ms = time.time() * 1000
    n = len(df)
    last_close = float(df["c"].iloc[-1])
    last_ot = int(df["ot"].iloc[-1])
    sn = st["snap"]
    if RUN_MODE == "test":
        send("ทดสอบบอท OK " + SYMBOL + " (Pine port)\nแหล่งข้อมูล " + src
             + "\nแท่งล่าสุด (เวลาไทย) " + th(last_ot)
             + "\nราคาปิด " + f2(last_close)
             + "\nสถานะไม้ล่าสุด " + st["status"] + " " + st["side"] + " " + st["type"]
             + "\nATR " + f2(sn["atr"]) + "  RSI " + f2(sn["rsi"]) + "  ADX " + f2(sn["adx"])
             + "\nBias " + sn["bias"] + "  S/R " + f2(sn["s1"]) + " / " + f2(sn["r1"])
             + "\nผลจำลองในข้อมูลนี้ " + str(st["win"]) + "W " + str(st["loss"]) + "L รวม " + format(st["net"], "+.1f") + " จุด")
        return
    if RUN_MODE == "compare":
        rows = [short_line(e) for e in events[-SHOW:]]
        head = "เทียบสัญญาณ " + SYMBOL + " M15 (เวลาไทย) แหล่งข้อมูล " + src + "\n"
        send(head + ("\n".join(rows) if rows else "ไม่มีเหตุการณ์ในช่วงข้อมูลนี้"))
        return
    last_sent = load_last()
    changed = False
    for e in events:
        if e["i"] < n - LOOKBACK or e["ot"] <= last_sent:
            continue
        age = (now_ms - (e["ot"] + BAR_MS - 1)) / 60000.0
        if age > MAX_AGE_MIN:
            continue
        txt = describe(e)
        if age > 16:
            txt += "\n(ล่าช้า " + str(int(age)) + " นาที ราคาล่าสุด " + f2(last_close) + ")"
            if e["kind"] in ("BUY", "SELL"):
                if e["kind"] == "BUY" and (last_close <= e["sl"] or last_close >= e["tp"]):
                    txt += "\nราคาผ่าน SL/TP แล้ว ไม่ควรเข้า"
                if e["kind"] == "SELL" and (last_close >= e["sl"] or last_close <= e["tp"]):
                    txt += "\nราคาผ่าน SL/TP แล้ว ไม่ควรเข้า"
        send(txt)
        last_sent = e["ot"]
        changed = True
    if changed:
        with open(STATE_FILE, "w") as f:
            json.dump({"last": last_sent}, f)


if __name__ == "__main__":
    main()
