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
        df = yf.download([s + ".NS" for s in b], period="3y", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for s in b:
            try:
                x = df[s + ".NS"].dropna(subset=["Close"])
                if len(x) > 150: data[s] = x[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except Exception:
                pass
        print(f"downloaded {min(i + 100, len(syms))}/{len(syms)}", flush=True)
    n = yf.download("^NSEI", period="3y", interval="1d", auto_adjust=True, progress=False)
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


# ---------------- MOMENTUM PORTFOLIO (v6, main strategy) ----------------
# Every WEEK (first trading day of the week, after the close) rotate into the 20 strongest liquid stocks ranked by
# 12-month return (skipping the last month) divided by volatility. Market switch with a 4% band: go to CASH when
# Nifty closes a review day 4% below its 200-day average, come back when it is 4% above (stops weekly flip-flopping).
# Holdings are kept while they stay in the top 40; about 2 stocks are swapped per week.
# Backtest 2012-Oct 2026 (all NSE stocks, 0.3% costs): 18.4% a year, worst fall -32%, losing years 2018 -20%, 2022 -10%,
# 2025 -10%. (Monthly rotation of the same list did better: 23%/yr, worst fall -23%.) Real results will be lower.
P_N, P_BUF, P_MINTURN, P_BAND = 20, 2, 10.0, 4.0   # stocks, keep-buffer, min turnover Rs cr, market band %


def pf_ranks(data):
    C = pd.DataFrame({s: d.Close for s, d in data.items()}).sort_index()
    H = pd.DataFrame({s: d.High for s, d in data.items()}).reindex(C.index)
    T = pd.DataFrame({s: d.Close * d.Volume for s, d in data.items()}).reindex(C.index).rolling(20, min_periods=15).mean()
    vol = C.pct_change(fill_method=None).rolling(126).std() * np.sqrt(250)
    m12 = C.shift(21) / C.shift(252) - 1
    ok = (T >= P_MINTURN * 1e7) & (C > 20) & (C > C.rolling(200).mean()) & (C / H.rolling(252, min_periods=200).max() >= 0.75) & vol.notna() & m12.notna()
    return (m12 / vol).where(ok).rank(axis=1, ascending=False, method="first"), m12


def pf_update(j, data, nif, PR, M12):
    """Monthly decision, stored permanently in the journal."""
    dstr = lambda ts: str(pd.Timestamp(ts).date())
    pf = j.setdefault("pf", {"holdings": [], "log": [], "last": None})
    if pf.get("freq") != "W":                     # switched from monthly to weekly: start a fresh record
        pf.clear(); pf.update(holdings=[], log=[], last=None, freq="W", state=None)
    nd = nif.Close; n200 = nd.rolling(200).mean()
    last = nd.index[-1]
    wk = nd.index.to_period("W")
    first = nd.index[wk == wk[-1]][0]
    if pf["last"] != dstr(first) and first in PR.index:
        x = (nd[first] / n200[first] - 1) * 100
        if pf.get("state") is None: pf["state"] = bool(x > 0)
        elif pf["state"] and x < -P_BAND: pf["state"] = False
        elif not pf["state"] and x > P_BAND: pf["state"] = True
        ok = pf["state"]
        r = PR.loc[first].dropna()
        old = {h["sym"]: h for h in pf["holdings"]}
        if ok:
            keep = [x for x in old if x in r.index and r[x] <= P_N * P_BUF]
            add = [x for x in r.sort_values().index if x not in keep][:P_N - len(keep)]
            new = keep + add
        else:
            new = []
        px = lambda x: r2(data[x].Close.get(first)) if x in data else None
        sells = [dict(sym=x, entry=old[x]["entry"], since=old[x]["since"], exit=px(x),
                      pnl=r2((px(x) / old[x]["entry"] - 1) * 100) if px(x) and old[x]["entry"] else None) for x in old if x not in new]
        buys = [dict(sym=x, price=px(x), rank=int(r[x])) for x in new if x not in old]
        pf["holdings"] = [old[x] for x in new if x in old] + [dict(sym=b["sym"], since=dstr(first), entry=b["price"]) for b in buys]
        pf["log"].insert(0, dict(date=dstr(first), mode="INVESTED" if ok else "CASH", nifty=r2(nd[first]), n200=r2(n200[first]), buys=buys, sells=sells))
        pf["last"] = dstr(first)
    hold = []
    for h in pf["holdings"]:
        d = data.get(h["sym"])
        now = r2(d.Close.iloc[-1]) if d is not None else None
        hold.append(dict(h, now=now, pnl=r2((now / h["entry"] - 1) * 100) if now and h["entry"] else None,
                         rank=int(PR[h["sym"]].iloc[-1]) if h["sym"] in PR and PR[h["sym"]].iloc[-1] == PR[h["sym"]].iloc[-1] else None,
                         m12=r2(M12[h["sym"]].iloc[-1] * 100) if h["sym"] in M12 else None))
    nl, n2 = float(nd.iloc[-1]), float(n200.iloc[-1])
    nxt = (pd.Timestamp(first) + pd.Timedelta(days=7)).strftime("%d %b %Y")
    return dict(mode="INVESTED" if pf["holdings"] else "CASH", decided=pf["last"], holdings=sorted(hold, key=lambda h: -(h["pnl"] or 0)),
                log=pf["log"][:30], nifty=round(nl, 1), n200=round(n2, 1), nifty_ok_now=nl > n2, next_review=f"first trading day of next week ({nxt} or the next open day)", band=P_BAND,
                state=pf.get("state"), gap200=round((nl / n2 - 1) * 100, 1),
                act_tomorrow=pf["last"] == dstr(last), n=P_N, minturn=P_MINTURN)


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


# ---------------- permanent trade journal (v5) ----------------
# Every signal is written ONCE into journal.enc (encrypted with APP_PASS, committed back to the repo) and then
# followed with its own fixed entry / target / stop until it closes. Rankings changing later never remove it.
JFILE, JSTART = "journal.enc", "2026-10-23"          # JSTART = first day the app was live


def _aes(salt):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    return AESGCM(PBKDF2HMAC(hashes.SHA256(), 32, salt, 150000).derive(os.environ.get("APP_PASS", "").encode()))


def load_journal():
    import base64
    if os.path.exists(JFILE):
        try:
            b = json.load(open(JFILE)); a = base64.b64decode(b["data"])
            return json.loads(_aes(base64.b64decode(b["salt"])).decrypt(a[:12], a[12:], None))
        except Exception as e:
            print("journal could not be opened (password changed?) - starting a new one:", e)
    return {"since": JSTART, "done_to": None, "trades": [], "leaders": {}}


def save_journal(j):
    import base64
    salt, iv = os.urandom(16), os.urandom(12)
    data = base64.b64encode(iv + _aes(salt).encrypt(iv, json.dumps(j, default=J).encode(), None)).decode()
    json.dump({"salt": base64.b64encode(salt).decode(), "data": data}, open(JFILE, "w"))


def j_update(e, d):
    """Follow one journal trade on its own fixed prices (same exit rules as the backtest)."""
    dstr = lambda ts: str(pd.Timestamp(ts).date())
    idx = d.index; O, H, C = d.Open.values, d.High.values, d.Close.values
    e["last"], e["last_date"] = r2(C[-1]), dstr(idx[-1])
    if e["status"] in ("target", "stop", "time", "skipped"):
        return
    k = int(idx.searchsorted(pd.Timestamp(e["sig_date"]), side="right"))
    if k >= len(C):
        e["status"] = "waiting"; return
    if e.get("entry") is None:
        if O[k] > e["sig_close"] * (1 + e["gap_pct"] / 100):
            e.update(status="skipped", entry_date=dstr(idx[k]), open_px=r2(O[k])); return
        e.update(entry=r2(O[k]), entry_date=dstr(idx[k]), target=r2(O[k] * (1 + e["tgt_pct"] / 100)), stop=r2(O[k] * (1 - e["stop_pct"] / 100)))
    k = int(idx.searchsorted(pd.Timestamp(e["entry_date"])))
    e.update(status="open", sell_next=False)
    for i in range(k, len(C)):
        if H[i] >= e["target"]:
            e.update(status="target", exit=r2(max(e["target"], O[i]) if i > k else e["target"]), exit_date=dstr(idx[i])); break
        if C[i] <= e["stop"]:
            if i + 1 < len(C): e.update(status="stop", exit=r2(O[i + 1]), exit_date=dstr(idx[i + 1]))
            else: e["sell_next"] = True
            break
        if i - k >= MAXD:
            e.update(status="time", exit=r2(C[i]), exit_date=dstr(idx[i])); break
    end = idx.searchsorted(pd.Timestamp(e["exit_date"])) + 1 if e.get("exit_date") else len(C)
    e["days"] = int(end - k)
    e["best"] = r2((H[k:end].max() / e["entry"] - 1) * 100)
    e["worst"] = r2((d.Low.values[k:end].min() / e["entry"] - 1) * 100)
    px = e.get("exit") if e["status"] in ("target", "stop", "time") else C[-1]
    e["pnl"] = r2((px / e["entry"] - 1) * 100 - (0.25 if e["status"] in ("target", "stop", "time") else 0))


def main():
    os.makedirs(OUT + "/c", exist_ok=True)
    syms, watch = load_universe()
    data, nif = download(syms)
    dstr = lambda ts: str(pd.Timestamp(ts).date())
    allrows, stats = [], {k: {"trades": 0, "wins": 0, "sum": 0.0} for k in ["LEADER", *SETUPS]}
    last_day = max(d.index[-1] for d in data.values())
    RK, M6 = leader_ranks(data)
    j = load_journal()
    done_to = pd.Timestamp(j["done_to"] or (pd.Timestamp(JSTART) - pd.Timedelta(days=1)))
    # 1) leader history: who was in the top list on every new day
    for day in RK.index[RK.index > done_to]:
        top = RK.loc[day].dropna(); top = top[top <= L_TOP]
        for s, v in top.items():
            px = r2(data[s].Close.get(day))
            L = j["leaders"].setdefault(s, {"first": dstr(day), "first_px": px, "days": 0, "best_rank": int(v)})
            L.update(last=dstr(day), days=L["days"] + 1, best_rank=min(L["best_rank"], int(v)), last_rank=int(v))
    # 2) signals: compute every stock, add each NEW signal to the journal once
    trades_by = {}
    for s, d in data.items():
        trs = sorted(engine(d, nif.Close) + leader_engine(d, RK[s]), key=lambda t: t["sig_i"]); trades_by[s] = trs
        for t in trs:
            sd = d.index[t["sig_i"]]
            if t["status"] not in ("buy_next_open", "open") and t.get("exit_i") is not None:
                r = t["exit"] / t["entry"] - 1 - 0.0025
                st = stats[t["setup"]]; st["trades"] += 1; st["wins"] += int(r > 0); st["sum"] += r
            if sd <= done_to: continue
            jid = f"{s}|{t['setup']}|{dstr(sd)}"
            if any(e["id"] == jid for e in j["trades"]): continue
            ld = t["setup"] == "LEADER"
            j["trades"].append(dict(id=jid, sym=s, setup=t["setup"], sig_date=dstr(sd), sig_close=r2(d.Close.iloc[t["sig_i"]]), rank=t.get("rank"),
                                    tgt_pct=L_TGT if ld else TGT, stop_pct=L_STOP if ld else STOP, gap_pct=L_GAP if ld else GAPMAX,
                                    entry=None, status="waiting"))
    for e in j["trades"]:
        if e["sym"] in data: j_update(e, data[e["sym"]])
    j["done_to"] = dstr(last_day)
    PR, M12 = pf_ranks(data)
    portfolio = pf_update(j, data, nif, PR, M12)
    save_journal(j)
    # 3) leaders list now + history of everyone who was a leader since the app started
    lr = RK.iloc[-1].dropna().sort_values().head(L_TOP)
    leaders = [dict(sym=s, rank=int(v), price=r2(data[s].Close.iloc[-1]), m6=round(float(M6[s].iloc[-1] * 100), 1),
                    off_hi=round(float((data[s].Close.iloc[-1] / data[s].High.tail(252).max() - 1) * 100), 1)) for s, v in lr.items()]
    lhist = []
    for s, L in j["leaders"].items():
        now = r2(data[s].Close.iloc[-1]) if s in data else None
        lhist.append(dict(sym=s, first=L["first"], first_px=L["first_px"], last=L["last"], days=L["days"], best_rank=L["best_rank"],
                          now=now, chg=r2((now / L["first_px"] - 1) * 100) if now and L["first_px"] else None,
                          in_top=L["last"] == dstr(last_day), rank=int(lr[s]) if s in lr else None))
    lhist.sort(key=lambda x: (not x["in_top"], x["rank"] or 999, -pd.Timestamp(x["last"]).value))
    # 4) chart files: journal trades drawn on the chart
    jby = {}
    for e in j["trades"]: jby.setdefault(e["sym"], []).append(e)
    for s, d in data.items():
        C = d.Close.values; markers, lines = [], []
        for e in jby.get(s, []):
            markers.append([e["sig_date"], "sig", e["setup"]])
            if e.get("entry") is not None: markers.append([e["entry_date"], "buy", e["entry"], e["setup"]])
            if e["status"] in ("target", "stop", "time"): markers.append([e["exit_date"], e["status"], e["pnl"]])
            if e["status"] == "open":
                lines = [{"v": e["target"], "c": "#00c853", "t": "Target"}, {"v": e["stop"], "c": "#ff5252", "t": "Stop"},
                         {"v": e["entry"], "c": "#2f81f7", "t": "Bought"}]
        tail = d.tail(300)
        cand = [[dstr(i), r2(r.Open), r2(r.High), r2(r.Low), r2(r.Close), int(r.Volume)] for i, r in tail.iterrows()]
        json.dump({"sym": s, "c": cand, "m": [m for m in markers if m[0] >= cand[0][0]], "l": lines}, open(f"{OUT}/c/{s}.json", "w"), separators=(",", ":"), default=J)
        chg = (C[-1] / C[-2] - 1) * 100 if len(C) > 1 else 0
        es = [e["status"] for e in jby.get(s, [])]
        state = "BUY" if "waiting" in es else ("OPEN" if "open" in es else "")
        allrows.append([s, r2(C[-1]), round(chg, 2), state])
    for k, v in stats.items():
        v["win_pct"] = round(100 * v["wins"] / v["trades"], 1) if v["trades"] else None
        v["avg_pct"] = round(100 * v["sum"] / v["trades"], 2) if v["trades"] else None
        del v["sum"]
    nl = float(nif.Close.iloc[-1]); nd = float(nif.Close.rolling(200).mean().iloc[-1]); n20 = float(nif.Close.iloc[-1] / nif.Close.iloc[-21] - 1) * 100
    ist = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)
    out = {"updated": ist.strftime("%d %b %Y, %I:%M %p IST"), "data_date": dstr(last_day), "universe": len(data),
           "nifty": {"last": round(nl, 1), "dma200": round(nd, 1), "r20": round(n20, 1), "ok": n20 > 0},
           "portfolio": portfolio, "journal": sorted(j["trades"], key=lambda e: e["sig_date"], reverse=True), "since": j["since"],
           "leaders": leaders, "leader_hist": lhist, "stats_2y": stats, "watch_default": watch,
           "rules": {"l_top": L_TOP, "l_tgt": L_TGT, "l_stop": L_STOP, "tr": TR, "r20": R20, "r20max": R20MAX, "target": TGT, "stop": STOP, "maxd": MAXD, "atr": ATRMAX, "ext": EXTMAX, "gap": GAPMAX}}
    json.dump(out, open(f"{OUT}/signals.json", "w"), separators=(",", ":"), default=J)
    json.dump(allrows, open(f"{OUT}/all.json", "w"), separators=(",", ":"), default=J)
    encrypt_site()
    n_w = sum(e["status"] == "waiting" for e in j["trades"]); n_o = sum(e["status"] == "open" for e in j["trades"])
    print(f"done: {len(data)} stocks, journal {len(j['trades'])} trades ({n_w} waiting, {n_o} open), {len(j['leaders'])} leaders tracked")


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
