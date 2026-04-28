import streamlit as st

from config import Config
from statarb.backtest.engine import BacktestResult


def get_config() -> Config | None:
    return st.session_state.get("config")


def set_config(config: Config):
    st.session_state["config"] = config


def get_backtest_result() -> BacktestResult | None:
    return st.session_state.get("backtest_result")


def set_backtest_result(result: BacktestResult):
    st.session_state["backtest_result"] = result


def has_backtest_result() -> bool:
    return "backtest_result" in st.session_state


def get_prices() -> "pd.DataFrame | None":
    return st.session_state.get("prices")


def set_prices(prices):
    st.session_state["prices"] = prices


def get_volume() -> "pd.DataFrame | None":
    return st.session_state.get("volume")


def set_volume(volume):
    st.session_state["volume"] = volume


def set_engine_inputs(returns, etf_returns, spy_returns, sector_mapping):
    """Cache the auxiliary frames the paper-faithful engine needs per day."""
    st.session_state["engine_returns"] = returns
    st.session_state["engine_etf_returns"] = etf_returns
    st.session_state["engine_spy_returns"] = spy_returns
    st.session_state["engine_sector_mapping"] = sector_mapping


def get_engine_inputs():
    return (
        st.session_state.get("engine_returns"),
        st.session_state.get("engine_etf_returns"),
        st.session_state.get("engine_spy_returns"),
        st.session_state.get("engine_sector_mapping"),
    )
