"""
Scanner that runs in the cloud (GitHub Actions) every trading day and writes the data the iPhone app shows.
Two setups, same engine (daily candles):
  IGNITION    – turnover surge + 60-day closing high + up >=10% in 20 days, within 10% of the 52-week high
  TURNAROUND  – same, but 10–20% BELOW the 52-week high (stock fell, then breaks out: the JINDALPOLY type)
Trade rules: buy next open, sell at +30% (target) or -15% (stop), max 120 trading days.
Backtest 2012–2026 (~1,000 NSE stocks, charges incl.): IGNITION ≈ 48% winners, +5.1%/trade;
TURNAROUND (−10…−20%) ≈ 45% winners, +3.6%/trade over 14 yrs (only ≈ break-even in 2024-26);
deeper falls (>20% below high) lost money in 2024-26 and >30% lost over 14 yrs, so they are excluded.
"""
import os, json, math, datetime as dt
import numpy as np, pandas as pd

OUT = "site/data"
SETUPS = {"IGNITION": (-10, 1000), "TURNAROUND": (-20, -10)}     # allowed % distance from 52-week high
TR, R20, TGT, STOP, MAXD, MINTURN_CR, COOL = 2.5, 10, 30, 15, 120, 1.0, 40


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


def engine(d):
    O, H, L, C, V = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
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
            if (setup and tr[i] >= TR and C[i] > hc60[i] and r20[i] * 100 >= R20 and t10[i] / 1e7 >= MINTURN_CR and i - last >= COOL):
                last = i
                t = dict(setup=setup, sig_i=i, tr=round(float(tr[i]), 1), r20=round(float(r20[i] * 100), 1), off_hi=round(float(off), 1))
                if i + 1 >= n:
                    t.update(entry_i=None, status="buy_next_open"); trades.append(t); break
                k = i + 1; e = O[k]
                cur = dict(t, entry_i=k, entry=e, target=e * (1 + TGT / 100), stop=e * (1 - STOP / 100), status="open")
                i = k; continue
            i += 1; continue
        if L[i] <= cur["stop"]:
            cur.update(exit_i=i, exit=min(cur["stop"], O[i]) if i > cur["entry_i"] else cur["stop"], status="stop"); trades.append(cur); cur = None
        elif H[i] >= cur["target"]:
            cur.update(exit_i=i, exit=max(cur["target"], O[i]) if i > cur["entry_i"] else cur["target"], status="target"); trades.append(cur); cur = None
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
    stats = {k: {"trades": 0, "wins": 0, "sum": 0.0} for k in SETUPS}
    last_day = max(d.index[-1] for d in data.values())
    for s, d in data.items():
        trs = engine(d); C = d.Close.values
        markers = []
        for t in trs:
            if t["status"] == "buy_next_open":
                buy.append(dict(sym=s, setup=t["setup"], price=r2(C[-1]), tr=t["tr"], r20=t["r20"], off_hi=t["off_hi"],
                                target=r2(C[-1] * (1 + TGT / 100)), stop=r2(C[-1] * (1 - STOP / 100)), date=dstr(d.index[-1])))
                markers.append([dstr(d.index[t["sig_i"]]), "sig", t["setup"]]); continue
            markers.append([dstr(d.index[t["entry_i"]]), "buy", round(t["entry"], 2), t["setup"]])
            if t["status"] == "open":
                opn.append(dict(sym=s, setup=t["setup"], entry=r2(t["entry"]), entry_date=dstr(d.index[t["entry_i"]]), last=r2(C[-1]),
                                pnl=r2((C[-1] / t["entry"] - 1) * 100), target=r2(t["target"]), stop=r2(t["stop"]),
                                days=len(C) - 1 - t["entry_i"]))
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
    nl = float(nif.Close.iloc[-1]); nd = float(nif.Close.rolling(200).mean().iloc[-1])
    ist = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)
    out = {"updated": ist.strftime("%d %b %Y, %I:%M %p IST"), "data_date": dstr(last_day), "universe": len(data),
           "nifty": {"last": round(nl, 1), "dma200": round(nd, 1), "ok": nl > nd},
           "buy": sorted(buy, key=lambda x: -x["tr"]), "open": sorted(opn, key=lambda x: -x["pnl"]),
           "recent": sorted(recent, key=lambda x: x["date"], reverse=True), "stats_2y": stats, "watch_default": watch,
           "rules": {"tr": TR, "r20": R20, "target": TGT, "stop": STOP, "maxd": MAXD}}
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
