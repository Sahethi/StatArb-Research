# StatArb-Research

A statistical arbitrage research platform implementing the Avellaneda & Lee (2010)
mean-reversion strategy on US equities, with extensions for regime gating,
volatility targeting, and pairs trading.

## What it does

Given a basket of tickers and a date range, the pipeline:

1. **Fetches** daily prices, volume, and returns (yfinance or CRSP/WRDS).
2. **Strips systematic factors** using one of four models:
   - `pca` — rolling PCA on the return covariance (eigenportfolios).
   - `etf` — per-sector ETF regression.
   - `combined` — SPY + sector ETF + residual PCA stack.
   - `pairs` — cointegrated pairs via Engle–Granger.
3. **Fits an Ornstein–Uhlenbeck process** to each residual return series and
   computes a normalized **s-score** = (X − m) / σ_eq.
4. **Generates long/short signals** when s-scores cross configurable open/close
   thresholds (defaults: s_bo=1.25, s_so=1.25, s_sc=0.50, s_bc=0.75).
5. **Backtests** with transaction costs, leverage limits, and SPY hedging.
6. **Reports** Sharpe, Sortino, drawdown, win rate, trade log, and diagnostics.

### Extensions

- **HMM regime gating** — 2-state Gaussian HMM fit on the rolling level and
  stability of cross-sectional residual volatility; skips new entries when
  the posterior probability of the favorable (mean-reverting) regime falls
  below a configurable threshold.
- **Vol targeting** — vol-parity position sizing using each name's OU
  equilibrium volatility (σ_eq), with a floor/cap to prevent degenerate
  notionals.
- **Almgren–Chriss** — optimal execution cost model. *Implemented in
  [statarb/extensions/almgren_chriss.py](statarb/extensions/almgren_chriss.py)
  but not yet wired into the backtest engine.*

## Quick start

### Local development (FastAPI + React, recommended)

```bash
# 1. Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000

# 2. Frontend (separate terminal)
cd frontend
npm install
npm run dev          # http://localhost:5173 — proxies /api to :8000
```

Open `http://localhost:5173` and configure a backtest from the UI.

### Streamlit app (alternative UI)

```bash
streamlit run app/Home.py
```

### Deploy to Railway

The repo ships with a multi-stage [Dockerfile](Dockerfile) and
[railway.json](railway.json). On Railway:

1. New Project → Deploy from GitHub repo → pick this repo.
2. Railway auto-detects the Dockerfile and builds (~5–8 min first time).
3. Settings → Networking → Generate Domain.

The FastAPI service serves both `/api/*` endpoints and the built React
frontend from the same origin.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/config/defaults` | Default tickers, data sources, model list |
| `GET /api/cache` | List backtest cache entries (size, mtime) |
| `DELETE /api/cache` | Clear the backtest cache |
| `POST /api/backtest` | Run a backtest, return metrics + curves |
| `POST /api/backtest/stream` | Same, as Server-Sent Events with progress |
| `POST /api/cointegration` | Engle–Granger cointegration scan |
| `POST /api/grid-search` | Sweep entry thresholds (s_bo × s_so) |

## Project structure

```
backend/         FastAPI orchestration layer (no trading logic)
frontend/        Vite + React + Tailwind UI
app/             Legacy Streamlit UI
statarb/
  data/          yfinance + CRSP data sources, sector mapping
  factors/       PCA, ETF, combined, pairs trading factor models
  signals/       OU estimator, s-score, cointegration, volume-time, eligibility filters
  backtest/      Engine, portfolio, costs, metrics
  extensions/    HMM regime, vol targeting, Almgren-Chriss
config.py        Centralized config + ticker universes
tests/           Unit tests
```

## Data sources

- **yfinance** (default) — free, no auth. Some delisted tickers return empty;
  filtered automatically.
- **CRSP via WRDS** — set `WRDS_USERNAME` env var; requires WRDS subscription.
  Necessary for the full 1997–2007 paper-replication universe (delisted names).

Preset universes in `config.py`:
- `PAPER_TICKERS` — 725 names approximating the Avellaneda–Lee 1997–2007
  CRSP universe (S&P 500 survivors + large-cap delisted names like Lehman,
  Bear Stearns, Compaq, Sun, WaMu).
- `DEFAULT_TICKERS` — currently aliased to `PAPER_TICKERS`
  ([config.py:202](config.py#L202)).
- `MODERN_TICKERS` — 699 modern large-caps spanning megacap tech, banks,
  industrials, and recent IPOs (ABNB, COIN, PLTR, etc.).

## Tech stack

Python 3.11, FastAPI, NumPy/Pandas/SciPy, scikit-learn, statsmodels, hmmlearn,
yfinance, WRDS · React 18, Vite, TypeScript, Tailwind, Recharts · Docker, Railway.

## References

- Avellaneda, M., & Lee, J. H. (2010). *Statistical arbitrage in the US equities
  market.* Quantitative Finance. (PDF in repo root.)
