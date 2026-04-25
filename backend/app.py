"""
FastAPI backend exposing the existing statarb pipeline as JSON endpoints.

This file ONLY orchestrates calls to the unchanged statarb.* and config.*
modules — no trading / signal / model logic lives here.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the project root importable so `import config`, `import statarb` work
# regardless of where uvicorn is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (  # noqa: E402
    Config, FactorConfig, OUConfig, SignalConfig, VolumeConfig,
    BacktestConfig, PairsConfig, DEFAULT_TICKERS, DATA_SOURCES, MARKET_ETF,
    PAPER_TICKERS, MODERN_TICKERS,
)
from statarb.data.universe import get_data_source, get_sector_mapping  # noqa: E402
from statarb.factors.registry import build_factor_model  # noqa: E402
from statarb.backtest.engine import run_backtest  # noqa: E402

app = FastAPI(title="StatArb API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    data_source: str = "yfinance"
    tickers: List[str]
    start_date: str
    end_date: str

    model_type: str = "pca"
    pca_lookback: int = 252
    pca_n_components: Optional[int] = 15
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252

    ou_window: int = 60
    kappa_min: float = 8.4
    mean_center: bool = True

    s_bo: float = 1.25
    s_so: float = 1.25
    s_sc: float = 0.50
    s_bc: float = 0.75
    s_limit: float = 4.0

    vol_enabled: bool = False
    vol_window: int = 10

    initial_equity: float = 1_000_000
    leverage_long: float = 2.0
    leverage_short: float = 2.0
    tc_bps: float = 1.0
    hedge_instrument: str = "SPY"

    pairs_pvalue: float = 0.05
    pairs_max: int = 20
    pairs_min_hl: float = 1.0
    pairs_max_hl: float = 126.0


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config/defaults")
def get_defaults():
    return {
        "default_tickers": DEFAULT_TICKERS,
        "paper_tickers_count": len(PAPER_TICKERS),
        "modern_tickers_count": len(MODERN_TICKERS),
        "data_sources": DATA_SOURCES,
        "model_types": [
            {"value": "pca", "label": "PCA (Eigenportfolios)"},
            {"value": "etf", "label": "Sector ETF Regression"},
            {"value": "combined", "label": "Combined (SPY + ETF + PCA)"},
            {"value": "pairs", "label": "Pairs Trading (Cointegration)"},
        ],
        "hedge_instruments": ["SPY", "sector_etf", "none"],
        "ticker_presets": {
            "default": DEFAULT_TICKERS,
            "paper": PAPER_TICKERS,
            "modern": MODERN_TICKERS,
        },
    }


def _build_config(req: BacktestRequest) -> Config:
    return Config(
        factor=FactorConfig(
            model_type=req.model_type,
            pca_lookback=req.pca_lookback,
            pca_n_components=req.pca_n_components,
            explained_variance_threshold=req.explained_variance_threshold,
            use_ledoit_wolf=req.use_ledoit_wolf,
            beta_rolling_window=req.beta_rolling_window,
        ),
        ou=OUConfig(
            estimation_window=req.ou_window,
            kappa_min=req.kappa_min,
            mean_center=req.mean_center,
        ),
        signal=SignalConfig(
            s_bo=req.s_bo, s_so=req.s_so, s_sc=req.s_sc,
            s_bc=req.s_bc, s_limit=req.s_limit,
        ),
        volume=VolumeConfig(
            enabled=req.vol_enabled, trailing_window=req.vol_window
        ),
        backtest=BacktestConfig(
            initial_equity=float(req.initial_equity),
            leverage_long=req.leverage_long,
            leverage_short=req.leverage_short,
            tc_bps=req.tc_bps,
            hedge_instrument=req.hedge_instrument,
        ),
        pairs=PairsConfig(
            pvalue_threshold=req.pairs_pvalue,
            max_pairs=req.pairs_max,
            min_half_life=req.pairs_min_hl,
            max_half_life=req.pairs_max_hl,
        ),
        data_source=req.data_source,
        start_date=req.start_date,
        end_date=req.end_date,
        tickers=req.tickers,
    )


@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        config = _build_config(req)

        data_source = get_data_source(config.data_source)
        all_tickers = config.tickers
        prices = data_source.fetch_prices(all_tickers, config.start_date, config.end_date)
        volume = data_source.fetch_volume(all_tickers, config.start_date, config.end_date)
        returns = data_source.fetch_returns(all_tickers, config.start_date, config.end_date)

        available = [t for t in all_tickers if t in prices.columns]
        prices = prices[available]
        volume = volume[[t for t in available if t in volume.columns]]
        returns = returns[[t for t in available if t in returns.columns]]

        sector_mapping = get_sector_mapping(available, data_source=data_source)
        factor_model = build_factor_model(
            config.factor, sector_mapping, pairs_cfg=config.pairs
        )

        kwargs: dict = {}
        etf_returns_df = None
        spy_returns_df = None
        needs_etf = (
            config.factor.model_type in ("etf", "combined")
            or config.backtest.hedge_instrument == "sector_etf"
        )
        needs_spy = (
            config.factor.model_type in ("combined", "pca")
            or config.backtest.hedge_instrument == "SPY"
        )

        if needs_etf:
            etf_tickers = list(set(sector_mapping.values()))
            etf_prices = data_source.fetch_prices(
                etf_tickers, config.start_date, config.end_date
            )
            etf_returns_df = np.log(etf_prices / etf_prices.shift(1)).dropna(how="all")
            kwargs["etf_returns"] = etf_returns_df
        if needs_spy:
            spy_prices = data_source.fetch_prices(
                [MARKET_ETF], config.start_date, config.end_date
            )
            spy_returns_df = np.log(spy_prices / spy_prices.shift(1)).dropna(how="all")
            kwargs["spy_returns"] = spy_returns_df
        if config.factor.model_type == "pairs":
            kwargs["prices"] = prices

        factor_result = factor_model.fit(returns, **kwargs)

        if config.factor.model_type == "pairs":
            pair_prices = {}
            for col in factor_result.residuals.columns:
                cs = factor_result.residuals[col].cumsum()
                first_finite = cs.dropna()
                if first_finite.empty:
                    continue
                pair_prices[col] = 100 * np.exp(cs - first_finite.iloc[0])
            bt_prices = pd.DataFrame(pair_prices)
            bt_volume = pd.DataFrame(
                np.ones(bt_prices.shape),
                index=bt_prices.index,
                columns=bt_prices.columns,
            )
            bt_returns = None
        else:
            bt_prices = prices
            bt_volume = volume
            bt_returns = returns

        result = run_backtest(
            config, bt_prices, bt_volume, factor_result,
            returns=bt_returns,
            etf_returns=etf_returns_df,
            spy_returns=spy_returns_df,
            sector_mapping=sector_mapping,
        )

        eq = result.equity_curve
        running_max = eq.cummax()
        drawdown = (eq / running_max - 1.0)

        yearly = eq.resample("YE").last()
        yearly_ret = yearly.pct_change().dropna()

        last_sscores = []
        if not result.daily_sscores.empty:
            last = result.daily_sscores.iloc[-1].dropna()
            for tk, val in last.items():
                if val <= -config.signal.s_bo:
                    sig = "LONG"
                elif val >= config.signal.s_so:
                    sig = "SHORT"
                else:
                    sig = "NEUTRAL"
                last_sscores.append({
                    "ticker": str(tk), "sscore": float(val), "signal": sig,
                })
            last_sscores.sort(key=lambda r: r["sscore"])

        exposure_curve = []
        if not result.daily_positions.empty:
            dp = result.daily_positions.copy()
            dp["long"] = np.where(dp["direction"] == 1, dp["notional"], 0.0)
            dp["short"] = np.where(dp["direction"] == -1, dp["notional"], 0.0)
            agg = dp.groupby("date").agg(
                long=("long", "sum"), short=("short", "sum")
            ).reset_index()
            for r in agg.itertuples():
                exposure_curve.append({
                    "date": str(r.date)[:10],
                    "long": float(r.long),
                    "short": float(r.short),
                })

        # Trades (cap to ~5000 for payload size).
        trades_list = []
        if not result.trades.empty:
            tr = result.trades.tail(5000)
            for r in tr.itertuples():
                trades_list.append({
                    "ticker": str(r.ticker),
                    "direction": int(r.direction),
                    "entry_date": str(r.entry_date)[:10],
                    "exit_date": str(r.exit_date)[:10],
                    "entry_price": float(r.entry_price),
                    "exit_price": float(r.exit_price),
                    "pnl": float(r.pnl),
                    "notional": float(r.notional),
                })

        # Factor diagnostics snapshot from the fit metadata.
        meta = factor_result.metadata or {}
        eigenvalues = meta.get("eigenvalues")
        all_eigenvalues = meta.get("all_eigenvalues")
        diag = {
            "model_type": config.factor.model_type,
            "eigenvalues": (
                [float(x) for x in eigenvalues[:20]]
                if eigenvalues is not None else []
            ),
            "all_eigenvalues_top": (
                [float(x) for x in all_eigenvalues[:50]]
                if all_eigenvalues is not None else []
            ),
            "explained_variance_ratio": float(
                meta.get("explained_variance_ratio") or 0.0
            ),
            "n_components": int(meta.get("n_components") or 0),
            "r_squared": [
                {"ticker": k, "r2": float(v)}
                for k, v in (meta.get("r_squared") or {}).items()
            ],
        }

        # Latest-day OU stats for a per-stock table.
        ou_rows = []
        if result.daily_ou_params:
            last_key = sorted(result.daily_ou_params.keys())[-1]
            for tk, p in result.daily_ou_params[last_key].items():
                ou_rows.append({
                    "ticker": str(tk),
                    "kappa": float(p.kappa),
                    "m": float(p.m),
                    "sigma_eq": float(p.sigma_eq),
                    "half_life": float(p.half_life),
                    "factor_beta": float(p.factor_beta),
                })

        m = result.metrics
        return {
            "metrics": {
                "total_return": float(m.total_return),
                "annualized_return": float(m.annualized_return),
                "annualized_vol": float(m.annualized_vol),
                "sharpe_ratio": float(m.sharpe_ratio),
                "sortino_ratio": float(m.sortino_ratio),
                "max_drawdown": float(m.max_drawdown),
                "win_rate": float(m.win_rate),
                "trade_win_rate": float(m.trade_win_rate),
                "profit_factor": float(m.profit_factor),
                "num_trades": int(m.num_trades),
                "total_costs": float(m.total_costs),
                "avg_holding_period": float(m.avg_holding_period),
            },
            "equity_curve": [
                {"date": str(d)[:10], "equity": float(v)} for d, v in eq.items()
            ],
            "drawdown_curve": [
                {"date": str(d)[:10], "drawdown": float(v)}
                for d, v in drawdown.items()
            ],
            "exposure_curve": exposure_curve,
            "annual_returns": [
                {"year": int(y.year), "return": float(v)}
                for y, v in yearly_ret.items()
            ],
            "last_sscores": last_sscores,
            "data_summary": {
                "n_requested": len(all_tickers),
                "n_returned": len(available),
                "n_dropped": len(all_tickers) - len(available),
            },
            "trades": trades_list,
            "diagnostics": diag,
            "ou_last": ou_rows,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Cointegration endpoint
# ─────────────────────────────────────────────────────────────────────────────
class CointRequest(BaseModel):
    data_source: str = "yfinance"
    tickers: List[str]
    start_date: str
    end_date: str
    pvalue_threshold: float = 0.05
    lookback: int = 252
    pair_t1: Optional[str] = None
    pair_t2: Optional[str] = None


@app.post("/api/cointegration")
def cointegration(req: CointRequest):
    try:
        from statarb.signals.cointegration import test_cointegration

        ds = get_data_source(req.data_source)
        prices = ds.fetch_prices(req.tickers, req.start_date, req.end_date)
        prices = prices.dropna(axis=1, how="all")

        coint_df = test_cointegration(
            prices, req.pvalue_threshold, lookback=req.lookback
        )
        rows = []
        for _, r in coint_df.iterrows():
            rows.append({
                "ticker1": r["ticker1"],
                "ticker2": r["ticker2"],
                "pvalue": float(r["pvalue"]),
                "score": float(r["score"]),
                "hedge_ratio": float(r["hedge_ratio"]),
                "spread_mean": float(r["spread_mean"]),
                "spread_std": float(r["spread_std"]),
                "half_life": (
                    float(r["half_life"]) if pd.notna(r["half_life"]) else None
                ),
            })

        spread = []
        if (req.pair_t1 and req.pair_t2
                and req.pair_t1 in prices.columns
                and req.pair_t2 in prices.columns):
            log_p = np.log(prices[[req.pair_t1, req.pair_t2]]).dropna()
            beta = float(np.polyfit(
                log_p[req.pair_t2].values, log_p[req.pair_t1].values, 1
            )[0])
            s = log_p[req.pair_t1] - beta * log_p[req.pair_t2]
            mean = float(s.mean())
            std = float(s.std())
            for d, v in s.items():
                spread.append({
                    "date": str(d)[:10],
                    "spread": float(v),
                    "z": (float(v) - mean) / std if std > 0 else 0.0,
                })

        return {"pairs": rows, "spread": spread}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Grid search endpoint (sweeps s_bo / s_so)
# ─────────────────────────────────────────────────────────────────────────────
class GridRequest(BacktestRequest):
    s_bo_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]
    s_so_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]


@app.post("/api/grid-search")
def grid_search(req: GridRequest):
    try:
        config = _build_config(req)

        ds = get_data_source(config.data_source)
        all_tickers = config.tickers
        prices = ds.fetch_prices(all_tickers, config.start_date, config.end_date)
        volume = ds.fetch_volume(all_tickers, config.start_date, config.end_date)
        returns = ds.fetch_returns(all_tickers, config.start_date, config.end_date)

        available = [t for t in all_tickers if t in prices.columns]
        prices = prices[available]
        volume = volume[[t for t in available if t in volume.columns]]
        returns = returns[[t for t in available if t in returns.columns]]

        sector_mapping = get_sector_mapping(available, data_source=ds)
        factor_model = build_factor_model(
            config.factor, sector_mapping, pairs_cfg=config.pairs
        )

        kwargs: dict = {}
        etf_returns_df = None
        spy_returns_df = None
        needs_etf = (
            config.factor.model_type in ("etf", "combined")
            or config.backtest.hedge_instrument == "sector_etf"
        )
        needs_spy = (
            config.factor.model_type in ("combined", "pca")
            or config.backtest.hedge_instrument == "SPY"
        )
        if needs_etf:
            etf_tickers = list(set(sector_mapping.values()))
            etf_prices = ds.fetch_prices(
                etf_tickers, config.start_date, config.end_date
            )
            etf_returns_df = np.log(etf_prices / etf_prices.shift(1)).dropna(how="all")
            kwargs["etf_returns"] = etf_returns_df
        if needs_spy:
            spy_prices = ds.fetch_prices(
                [MARKET_ETF], config.start_date, config.end_date
            )
            spy_returns_df = np.log(spy_prices / spy_prices.shift(1)).dropna(how="all")
            kwargs["spy_returns"] = spy_returns_df
        if config.factor.model_type == "pairs":
            kwargs["prices"] = prices

        factor_result = factor_model.fit(returns, **kwargs)

        if config.factor.model_type == "pairs":
            pair_prices = {}
            for col in factor_result.residuals.columns:
                cs = factor_result.residuals[col].cumsum()
                first_finite = cs.dropna()
                if first_finite.empty:
                    continue
                pair_prices[col] = 100 * np.exp(cs - first_finite.iloc[0])
            bt_prices = pd.DataFrame(pair_prices)
            bt_volume = pd.DataFrame(
                np.ones(bt_prices.shape),
                index=bt_prices.index,
                columns=bt_prices.columns,
            )
            bt_returns = None
        else:
            bt_prices = prices
            bt_volume = volume
            bt_returns = returns

        cells = []
        import copy as _copy
        for s_bo in req.s_bo_values:
            for s_so in req.s_so_values:
                cfg = _copy.deepcopy(config)
                cfg.signal.s_bo = float(s_bo)
                cfg.signal.s_so = float(s_so)
                res = run_backtest(
                    cfg, bt_prices, bt_volume, factor_result,
                    returns=bt_returns,
                    etf_returns=etf_returns_df,
                    spy_returns=spy_returns_df,
                    sector_mapping=sector_mapping,
                )
                cells.append({
                    "s_bo": float(s_bo),
                    "s_so": float(s_so),
                    "sharpe": float(res.metrics.sharpe_ratio),
                    "total_return": float(res.metrics.total_return),
                    "max_drawdown": float(res.metrics.max_drawdown),
                    "num_trades": int(res.metrics.num_trades),
                })

        # Best by Sharpe.
        best = max(cells, key=lambda c: c["sharpe"]) if cells else None
        return {
            "s_bo_values": req.s_bo_values,
            "s_so_values": req.s_so_values,
            "cells": cells,
            "best": best,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
