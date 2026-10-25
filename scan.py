"""
Scanner that runs in the cloud (GitHub Actions) every trading day and writes the data the iPhone app shows.
Two setups, same engine (daily candles):
  IGNITION    – turnover surge + 60-day closing high + up 10–25% in 20 days, within 10% of the 52-week high
  TURNAROUND  – same, but 10–20% BELOW the 52-week high (stock fell, then breaks out: the JINDALPOLY type)
Quality filters (v2, added after the Sep–Oct 2026 stop-outs):
  * Nifty must be UP over the last 20 days (no buying into a falling market)
  * not stretched: close <= 20% above its 20-day average, daily range (ATR20) <= 4.5% of price
  * skip the trade if it opens more than 2% above the signal close (don't chase gaps)
Trade rules: buy next open, sell at +25% (target), or if a day CLOSES 20% below entry sell next open
(intraday wicks no longer stop you out), max 120 trading days.
LEADER (v4) – top-30 stock by risk-adjusted 6m+3m momentum makes a 20-day closing high, within 8% of its 52w high;
  skip if it opens >3% up. Works in weak markets too. +10% target / sell if a day closes 30% below / 120 days.
  Backtest win rate: 2012-19 79% (+3.2%/trade), 2020-23 84% (+4.4%), 2024-26 77% (+1.7%), 2026 89% (+5.8%).
  Losers average about -20%, so one loss cancels about two wins.
Breakout setups backtest (~1,300 NSE stocks, charges incl.): 2012-19 59% win +5.4%/trade, 2020-26 64% win +7.9%/trade,
2024-26 54% win +3.9%/trade (old v1 rules: 40% win, +1.9%/trade in 2024-26).
"""
import os, json, math, datetime as dt
import numpy as np, pandas as pd

OUT = "site/data"
SETUPS = {"IGNITION": (-10, 1000), "TURNAROUND": (-20, -10)}     # allowed % distance from 52-week high
TR, R20, R20MAX, TGT, STOP, MAXD, MINTURN_CR, COOL = 2.5, 10, 25, 25, 20, 120, 1.0, 40
ATRMAX, EXTMAX, GAPMAX = 4.5, 20, 2   # % limits
# LEADER setup (v3): the stocks that survive falls = smooth, persistent relative strength
L_TOP, L_TGT, L_STOP, L_HI, L_GAP, L_COOL = 30, 10, 30, -8, 3, 20   # v4: tuned for >=75% win rate


def load_universe():
    u = open("universe.txt").read().split()
    w = open("watchlist.txt").read().split() if os.path.exists("watchlist.txt") else []
    return list(dict.fromkeys([x.strip().upper() for x in u + w if x.strip()])), [x.strip().upper() for x in w if x.strip()]


def download(syms):
    import yfinance as yf, logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data = {}
    for i in range(0, len(syms), 100):
        b = syms[i:i + 100]
        df = yf.download([s + ".NS" for s in b], period="2y", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for s in b:
            try:
                x = df[s + ".NS"].dropna(subset=["Close"])
                if len(x) > 150: data[s] = x[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
        print(f"downloaded {min(i + 100, len(syms))}/{len(syms)}", flush=True)
    n = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(n.columns, pd.MultiIndex): n = n.droplevel(1, axis=1)
    return data, n.dropna(subset=["Close"])


def engine(d, nifty=None):
    O, H, L, C, V = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    nr = (nifty / nifty.shift(20) - 1).reindex(d.index, method="ffill").values if nifty is not None else np.ones(len(C))
    atr = pd.Series(H - L).rolling(20).mean().values / C * 100
    ext = (C / pd.Series(C).rolling(20).mean().values - 1) * 100
    n = len(C); T = C * V
    t10 = pd.Series(T).rolling(10).mean().values
    t100 = pd.Series(T).rolling(100).mean().shift(10).values
    tr = t10 / np.where(t100 > 0, t100, np.nan)
    hc60 = pd.Series(C).rolling(60).max().shift(1).values
    hi252 = pd.Series(H).rolling(252, min_periods=120).max().values
    r20 = C / pd.Series(C).shift(20).values - 1
    trades, cur, last, i = [], None, -999, 120
    while i < n:
        if cur is None:
            off = (C[i] / hi252[i] - 1) * 100 if hi252[i] == hi252[i] else np.nan
            setup = next((k for k, (lo, hi) in SETUPS.items() if lo < off <= hi), None)
            if (setup and tr[i] >= TR and C[i] > hc60[i] and R20 <= r20[i] * 100 <= R20MAX and t10[i] / 1e7 >= MINTURN_CR and i - last >= COOL
                    and atr[i] <= ATRMAX and ext[i] <= EXTMAX and nr[i] > 0):
                last = i
                t = dict(setup=setup, sig_i=i, tr=round(float(tr[i]), 1), r20=round(float(r20[i] * 100), 1), off_hi=round(float(off), 1))
                if i + 1 >= n:
                    t.update(entry_i=None, status="buy_next_open"); trades.append(t); break
                k = i + 1; e = O[k]
                if e > C[i] * (1 + GAPMAX / 100):      # opened too high - skip, don't chase
                    i = k; continue
                cur = dict(t, entry_i=k, entry=e, target=e * (1 + TGT / 100), stop=e * (1 - STOP / 100), status="open")
                i = k; continue
            i += 1; continue
        if H[i] >= cur["target"]:
            cur.update(exit_i=i, exit=max(cur["target"], O[i]) if i > cur["entry_i"] else cur["target"], status="target"); trades.append(cur); cur = None
        elif C[i] <= cur["stop"]:                 # closing-basis stop: sell at next open
            if i + 1 >= n:
                cur["sell_next"] = True; break
            cur.update(exit_i=i + 1, exit=O[i + 1], status="stop"); trades.append(cur); cur = None; i += 2; continue
        elif i - cur["entry_i"] >= MAXD:
            cur.update(exit_i=i, exit=C[i], status="time"); trades.append(cur); cur = None
        i += 1
    if cur is not None: trades.append(cur)
    return trades


def leader_ranks(data):
    """Daily rank (1 = strongest) by risk-adjusted 6m + 3m momentum, among liquid stocks above 200DMA and within 25% of 52w high."""
    C = pd.DataFrame({s: d.Close for s, d in data.items()}).sort_index()
    H = pd.DataFrame({s: d.High for s, d in data.items()}).reindex(C.index)
    T = pd.DataFrame({s: d.Close * d.Volume for s, d in data.items()}).reindex(C.index).rolling(20, min_periods=15).mean()
    vol = C.pct_change(fill_method=None).rolling(63).std() * np.sqrt(250)
    m6, m3 = C / C.shift(126) - 1, C / C.shift(63) - 1
    ok = (T >= MINTURN_CR * 1e7) & (C > 20) & (C > C.rolling(200).mean()) & (C / H.rolling(252, min_periods=200).max() >= 0.75) & (vol >= 0.15) & m6.notna()
    sc = (m6 / vol).where(ok).rank(axis=1, ascending=False) + (m3 / vol).where(ok).rank(axis=1, ascending=False)
    return sc.rank(axis=1, method="first"), m6


def leader_engine(d, rk):
    O, H, L, C = (d[k].values for k in ("Open", "High", "Low", "Close"))
    n = len(C); r = rk.reindex(d.index).values
    hi20 = C >= pd.Series(C).rolling(20).max().values
    offhi = (C / pd.Series(H).rolling(252, min_periods=200).max().values - 1) * 100
    trades, cur, last, i = [], None, -999, 200
    while i < n:
        if cur is None:
            if r[i] == r[i] and r[i] <= L_TOP and hi20[i] and offhi[i] >= L_HI and i - last >= L_COOL:
                last = i
                t = dict(setup="LEADER", sig_i=i, tr=None, r20=round(float((C[i] / C[i - 20] - 1) * 100), 1), off_hi=None, rank=int(r[i]))
                if i + 1 >= n:
                    t.update(entry_i=None, status="buy_next_open"); trades.append(t); break
                k = i + 1; e = O[k]
                if e > C[i] * (1 + L_GAP / 100):
                    i = k; continue
                cur = dict(t, entry_i=k, entry=e, target=e * (1 + L_TGT / 100), stop=e * (1 - L_STOP / 100), status="open")
                i = k; continue
            i += 1; continue
        if H[i] >= cur["target"]:
            cur.update(exit_i=i, exit=max(cur["target"], O[i]) if i > cur["entry_i"] else cur["target"], status="target"); trades.append(cur); cur = None
        elif C[i] <= cur["stop"]:
            if i + 1 >= n:
                cur["sell_next"] = True; break
            cur.update(exit_i=i + 1, exit=O[i + 1], status="stop"); trades.append(cur); cur = None; i += 2; continue
        elif i - cur["entry_i"] >= MAXD:
            cur.update(exit_i=i, exit=C[i], status="time"); trades.append(cur); cur = None
        i += 1
    if cur is not None: trades.append(cur)
    return trades


def J(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError(type(o))


def r2(x): return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 2)


def main():
    os.makedirs(OUT + "/c", exist_ok=True)
    syms, watch = load_universe()
    data, nif = download(syms)
    dstr = lambda ts: str(pd.Timestamp(ts).date())
    buy, opn, recent, allrows = [], [], [], []
    stats = {k: {"trades": 0, "wins": 0, "sum": 0.0} for k in ["LEADER", *SETUPS]}
    last_day = max(d.index[-1] for d in data.values())
    RK, M6 = leader_ranks(data)
    lr = RK.iloc[-1].dropna().sort_values().head(L_TOP)
    leaders = [dict(sym=s, rank=int(v), price=r2(data[s].Close.iloc[-1]), m6=round(float(M6[s].iloc[-1] * 100), 1),
                    off_hi=round(float((data[s].Close.iloc[-1] / data[s].High.tail(252).max() - 1) * 100), 1)) for s, v in lr.items()]
    for s, d in data.items():
        trs = sorted(engine(d, nif.Close) + leader_engine(d, RK[s]), key=lambda t: t["sig_i"]); C = d.Close.values
        markers = []
        for t in trs:
            if t["status"] == "buy_next_open":
                ld = t["setup"] == "LEADER"; tg, sp = (L_TGT, L_STOP) if ld else (TGT, STOP)
                buy.append(dict(sym=s, setup=t["setup"], price=r2(C[-1]), tr=t["tr"], r20=t["r20"], off_hi=t["off_hi"], rank=t.get("rank"),
                                tgt_pct=tg, stop_pct=sp, target=r2(C[-1] * (1 + tg / 100)), stop=r2(C[-1] * (1 - sp / 100)),
                                max_entry=r2(C[-1] * (1 + (L_GAP if ld else GAPMAX) / 100)),
                                date=dstr(d.index[-1])))
                markers.append([dstr(d.index[t["sig_i"]]), "sig", t["setup"]]); continue
            markers.append([dstr(d.index[t["entry_i"]]), "buy", round(t["entry"], 2), t["setup"]])
            if t["status"] == "open":
                opn.append(dict(sym=s, setup=t["setup"], entry=r2(t["entry"]), entry_date=dstr(d.index[t["entry_i"]]), last=r2(C[-1]),
                                pnl=r2((C[-1] / t["entry"] - 1) * 100), target=r2(t["target"]), stop=r2(t["stop"]),
                                days=len(C) - 1 - t["entry_i"], sell=bool(t.get("sell_next"))))
            else:
                r = t["exit"] / t["entry"] - 1 - 0.0025
                st = stats[t["setup"]]; st["trades"] += 1; st["wins"] += int(r > 0); st["sum"] += r
                markers.append([dstr(d.index[t["exit_i"]]), t["status"], round(r * 100, 1)])
                if (last_day - d.index[t["exit_i"]]).days <= 30:
                    recent.append(dict(sym=s, setup=t["setup"], result=t["status"], pnl=round(r * 100, 1), date=dstr(d.index[t["exit_i"]])))
        # chart file (last 300 days)
        tail = d.tail(300)
        cand = [[dstr(i), r2(r.Open), r2(r.High), r2(r.Low), r2(r.Close), int(r.Volume)] for i, r in tail.iterrows()]
        lines = []
        cur = next((t for t in trs if t["status"] == "open"), None)
        if cur: lines = [{"v": r2(cur["target"]), "c": "#00c853", "t": "Target"}, {"v": r2(cur["stop"]), "c": "#ff5252", "t": "Stop"}]
        json.dump({"sym": s, "c": cand, "m": [m for m in markers if m[0] >= cand[0][0]], "l": lines}, open(f"{OUT}/c/{s}.json", "w"), separators=(",", ":"), default=J)
        chg = (C[-1] / C[-2] - 1) * 100 if len(C) > 1 else 0
        state = "BUY" if any(t["status"] == "buy_next_open" for t in trs) else ("OPEN" if cur else "")
        allrows.append([s, r2(C[-1]), round(chg, 2), state])
    for k, v in stats.items():
        v["win_pct"] = round(100 * v["wins"] / v["trades"], 1) if v["trades"] else None
        v["avg_pct"] = round(100 * v["sum"] / v["trades"], 2) if v["trades"] else None
        del v["sum"]
    nl = float(nif.Close.iloc[-1]); nd = float(nif.Close.rolling(200).mean().iloc[-1]); n20 = float(nif.Close.iloc[-1] / nif.Close.iloc[-21] - 1) * 100
    ist = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)
    out = {"updated": ist.strftime("%d %b %Y, %I:%M %p IST"), "data_date": dstr(last_day), "universe": len(data),
           "nifty": {"last": round(nl, 1), "dma200": round(nd, 1), "r20": round(n20, 1), "ok": n20 > 0},
           "buy": sorted(buy, key=lambda x: (x["setup"] != "LEADER", -(x["tr"] or 0))), "leaders": leaders, "open": sorted(opn, key=lambda x: (not x["sell"], -x["pnl"])),
           "recent": sorted(recent, key=lambda x: x["date"], reverse=True), "stats_2y": stats, "watch_default": watch,
           "rules": {"l_top": L_TOP, "l_tgt": L_TGT, "l_stop": L_STOP, "tr": TR, "r20": R20, "r20max": R20MAX, "target": TGT, "stop": STOP, "maxd": MAXD, "atr": ATRMAX, "ext": EXTMAX, "gap": GAPMAX}}
    json.dump(out, open(f"{OUT}/signals.json", "w"), separators=(",", ":"), default=J)
    json.dump(allrows, open(f"{OUT}/all.json", "w"), separators=(",", ":"), default=J)
    encrypt_site()
    print(f"done: {len(data)} stocks, {len(buy)} buy signals, {len(opn)} open trades")


def encrypt_site():
    """Lock every data file with the APP_PASS secret (AES-256-GCM, key = PBKDF2-SHA256).
    Refuses to publish anything if the secret is missing, so data is never public."""
    import base64, glob
    pw = os.environ.get("APP_PASS", "")
    if len(pw) < 6:
        raise SystemExit("APP_PASS secret missing/too short - add it in GitHub Settings > Secrets > Actions")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    salt = os.urandom(16)
    key = PBKDF2HMAC(hashes.SHA256(), 32, salt, 150000).derive(pw.encode())
    g = AESGCM(key)
    def enc(b):
        iv = os.urandom(12); return base64.b64encode(iv + g.encrypt(iv, b, None)).decode()
    for f in glob.glob(f"{OUT}/**/*.json", recursive=True):
        b = open(f, "rb").read(); open(f, "w").write(enc(b))
    json.dump({"salt": base64.b64encode(salt).decode(), "iter": 150000, "check": enc(b'"ok"')}, open(f"{OUT}/lock.json", "w"))


if __name__ == "__main__":
    main()
