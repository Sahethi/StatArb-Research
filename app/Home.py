import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from config import Config, SECTOR_ETFS, MARKET_ETF
from statarb.data.universe import get_data_source, get_sector_mapping
from statarb.factors.registry import build_factor_model
from statarb.backtest.engine import run_backtest
from app.state import (
    set_config, set_backtest_result, get_backtest_result, has_backtest_result,
    set_prices, set_volume, set_engine_inputs,
)
from app.components.sidebar import build_sidebar
from app.components.kpi_cards import render_kpi_cards
from app.components.df_display import show_df
from app.components.charts import (
    plot_equity_curve, plot_drawdown, plot_gross_exposure,
    plot_sscore_timeseries,
)


st.set_page_config(
    page_title="StatArb Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Statistical Arbitrage Dashboard")
st.caption("Avellaneda & Lee (2010) — PCA & ETF Mean-Reversion Strategies")

config = build_sidebar()
set_config(config)

if st.sidebar.button("Run Backtest", type="primary", width="stretch"):
    with st.spinner("Fetching data..."):
        data_source = get_data_source(config.data_source)

        all_tickers = config.tickers
        prices = data_source.fetch_prices(all_tickers, config.start_date, config.end_date)
        volume = data_source.fetch_volume(all_tickers, config.start_date, config.end_date)
        returns = data_source.fetch_returns(all_tickers, config.start_date, config.end_date)

        available = [t for t in all_tickers if t in prices.columns]
        prices = prices[available]
        volume = volume[[t for t in available if t in volume.columns]]
        returns = returns[[t for t in available if t in returns.columns]]

        # Post-download data-quality diagnostic. `yf.download` silently
        # returns columns only for tickers that had ANY data in the window,
        # so comparing requested vs. received tells you how many tickers
        # are effectively dropped (delisted names that don't exist on the
        # provider, invalid symbols, etc.).
        n_req = len(all_tickers)
        n_got = len(available)
        missing = [t for t in all_tickers if t not in prices.columns]
        coverage = returns.notna().sum()
        n_days = len(returns)
        full = int((coverage >= n_days * 0.95).sum()) if n_days else 0
        partial = int(((coverage > 0) & (coverage < n_days * 0.95)).sum()) if n_days else 0
        med_cov = int(coverage.median()) if n_got else 0

        print("=" * 60)
        print(f"[data] source={config.data_source}  window={config.start_date} → {config.end_date} ({n_days} trading days)")
        print(f"[data] tickers requested: {n_req}   returned: {n_got}   dropped: {n_req - n_got}")
        print(f"[data]   full coverage (≥95%): {full}")
        print(f"[data]   partial (IPO/delist mid-sample): {partial}")
        print(f"[data]   median days of data per surviving ticker: {med_cov}")
        if missing:
            preview = ", ".join(missing[:15])
            more = f"  … (+{len(missing) - 15} more)" if len(missing) > 15 else ""
            print(f"[data] no data returned for {len(missing)} tickers: {preview}{more}")
        print("=" * 60)

        st.info(
            f"**Data:** {n_got}/{n_req} tickers returned data  •  "
            f"{full} full coverage  •  {partial} partial  •  "
            f"{n_req - n_got} dropped"
        )

        set_prices(prices)
        set_volume(volume)

    with st.spinner("Computing sector mappings..."):
        # Uses TICKER_TO_ETF_OVERRIDES first, then the active data source
        # (yfinance .info for yfinance runs, CRSP SIC codes for CRSP runs),
        # then a final "XLY" fallback. No more yfinance-only dependency.
        sector_mapping = get_sector_mapping(available, data_source=data_source)

    with st.spinner("Fitting factor model..."):
        factor_model = build_factor_model(config.factor, sector_mapping, pairs_cfg=config.pairs)

        kwargs = {}
        etf_returns_df = None
        spy_returns_df = None

        # The engine's paper-faithful signal path runs a fresh 60-day OLS
        # per day, so it needs the factor-return frames directly — not just
        # the pre-computed residuals from `factor_model.fit`. Fetch ETF and
        # SPY returns whenever the hedge or the model needs them.
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

    with st.spinner("Running backtest..."):
        if config.factor.model_type == "pairs":
            pair_prices = {}
            for col in factor_result.residuals.columns:
                # cumsum skips NaN by default, so leading NaN rows (the
                # formation period before rolling β kicks in) stay NaN.
                cs = factor_result.residuals[col].cumsum()
                first_finite = cs.dropna()
                if first_finite.empty:
                    continue
                pair_prices[col] = 100 * np.exp(cs - first_finite.iloc[0])
            bt_prices = pd.DataFrame(pair_prices)
            bt_volume = pd.DataFrame(
                np.ones(bt_prices.shape), index=bt_prices.index, columns=bt_prices.columns
            )
        else:
            bt_prices = prices
            bt_volume = volume

        # For the pairs path, the universe traded is the SYNTHETIC pair
        # series ("T1_T2" columns), not the underlying stocks. The engine
        # must derive returns from `bt_prices` (the synthetic frame) so
        # `returns.columns` matches `prices.columns`. Passing the original
        # stock-returns frame here produces an empty intersection and
        # collapses every diagnostic to NaN.
        bt_returns = None if config.factor.model_type == "pairs" else returns
        result = run_backtest(
            config,
            bt_prices,
            bt_volume,
            factor_result,
            returns=bt_returns,
            etf_returns=etf_returns_df,
            spy_returns=spy_returns_df,
            sector_mapping=sector_mapping,
        )
        set_backtest_result(result)
        set_engine_inputs(returns, etf_returns_df, spy_returns_df, sector_mapping)

    st.success(
        f"Backtest complete: {result.metrics.num_trades} trades, "
        f"Sharpe = {result.metrics.sharpe_ratio:.2f}"
    )

if has_backtest_result():
    result = get_backtest_result()

    st.subheader("Performance Summary")
    render_kpi_cards(result.metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_equity_curve(result.equity_curve),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            plot_drawdown(result.equity_curve),
            width="stretch",
        )

    st.plotly_chart(
        plot_gross_exposure(result.daily_positions, result.equity_curve),
        width="stretch",
    )

    st.subheader("Current S-Scores")
    if not result.daily_sscores.empty:
        last_sscores = result.daily_sscores.iloc[-1].dropna().sort_values()
        sscore_df = pd.DataFrame({
            "Ticker": last_sscores.index,
            "S-Score": last_sscores.values,
            "Signal": [
                "LONG" if s <= -config.signal.s_bo
                else "SHORT" if s >= config.signal.s_so
                else "NEUTRAL"
                for s in last_sscores.values
            ],
        })

        def color_signal(val):
            if val == "LONG":
                return "background-color: rgba(44, 160, 44, 0.3)"
            elif val == "SHORT":
                return "background-color: rgba(214, 39, 40, 0.3)"
            return ""

        show_df(
            sscore_df.style.map(color_signal, subset=["Signal"]),
            height=400,
        )

    st.subheader("Per-Ticker Drill-Down")
    if not result.daily_sscores.empty:
        available_tickers = sorted(result.daily_sscores.columns.tolist())
        selected_ticker = st.selectbox("Select Ticker", available_tickers)

        if selected_ticker and selected_ticker in result.daily_sscores.columns:
            ticker_sscores = result.daily_sscores[selected_ticker].dropna()
            if not ticker_sscores.empty:
                st.plotly_chart(
                    plot_sscore_timeseries(
                        ticker_sscores, selected_ticker, config.signal
                    ),
                    width="stretch",
                )

            if not result.trades.empty:
                ticker_trades = result.trades[
                    result.trades["ticker"] == selected_ticker
                ]
                if not ticker_trades.empty:
                    st.write(f"**Trades for {selected_ticker}:**")
                    show_df(ticker_trades)

    st.subheader("Annual Performance")
    eq = result.equity_curve
    yearly = eq.resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()
    if not yearly_ret.empty:
        ann_df = pd.DataFrame({
            "Year": yearly_ret.index.year,
            "Return": [f"{r:.1%}" for r in yearly_ret.values],
        })
        show_df(ann_df)
else:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to begin.")
