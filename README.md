# Breakouts – iPhone app (free, works without your laptop)

GitHub runs the scanner in the cloud every weekday at 4:15 PM IST and publishes the app to a free web address. You add it to your iPhone home screen once; after that it opens full-screen like a normal app.

## One-time setup (about 10 minutes, can be done on a phone or PC)

1. Go to **github.com** and create a free account (for example, username `sk-trader`).
2. Click **+** (top right) → **New repository**.
   - Name: `breakouts`
   - Choose **Public** (free Pages hosting needs public).
   - Click **Create repository**.
3. Click **uploading an existing file**. Drag in **everything inside this `iphone_app` folder**: `index.html`, `scan.py`, `manifest.json`, `sw.js`, `icon-180.png`, `icon-512.png`, `universe.txt`, `watchlist.txt`, `README.md`. Then click **Commit changes**.
4. The workflow file sits in a hidden folder, so add it by hand:
   - Click **Add file → Create new file**.
   - Name it exactly `.github/workflows/scan.yml`.
   - Paste the contents of `.github/workflows/scan.yml` from this folder.
   - Click **Commit changes**.
5. Go to **Settings → Pages**. Under *Build and deployment → Source*, choose **GitHub Actions**.
6. Go to **Actions** → **Daily scan and publish app** → **Run workflow**. It takes 5–10 minutes the first time and should finish with a green ✓.
7. On your iPhone, open **Safari** (it must be Safari) and go to `https://YOUR-USERNAME.github.io/breakouts/`.
8. Tap **Share ⬆️ → Add to Home Screen → Add**. The "Breakouts" icon is now on your home screen.

After that it refreshes by itself every weekday evening and morning. To refresh immediately, use Actions → Run workflow (the GitHub iPhone app can do this too).

## Changing the stock list

- `universe.txt` is the scanned list (1,338 liquid NSE stocks). Edit it on GitHub to add or remove symbols.
- `watchlist.txt` is only the starting watchlist. On the phone, tap ☆ on any chart to add or remove stocks.

## What the app shows

- **Signals:** stocks to buy at tomorrow's open, with target, stop and quantity for ₹15k. There are two types:
  - 🔄 **Turnaround (the JINDALPOLY type):** the stock fell and is still 10–20% below its 52-week high, then turnover explodes (≥2.5× normal), it makes a 60-day closing high and is up ≥10% in 20 days.
  - 🚀 **Ignition:** the same surge, but within 10% of the 52-week high.
- **Open:** trades the rules are still holding, with live P&L, day count, target and stop.
- **Watch / Search:** any of the 1,338 stocks, with a chart showing signal, buy, target and stop markers, turnover bars, and the 50 and 200 DMAs.
- **Rules:** the exact rules and honest notes.

## Honest numbers

These are backtests using this app's own rules: +30% target, −15% stop, 120 days max, 0.25% charges.

| Setup | 2012–2026 | Last 2 years |
|---|---|---|
| 🚀 Ignition | ≈48% win, ≈+5%/trade | 45% win, +4.2%/trade |
| 🔄 Turnaround (−10…−20%) | ≈45% win, +3.6%/trade | 36% win, ≈+0.3%/trade |

- Signals deeper than 20% below the high (−20…−30%) lost money in 2024–26, so they are filtered out.
- Fewer than half of the trades win. The profit comes from winners being twice the size of losers.
- Data is end-of-day from Yahoo. This is not investment advice.
