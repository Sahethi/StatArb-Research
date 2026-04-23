"""
Portfolio management: position tracking, leverage, and beta-neutral hedging.

Implements the bang-bang position sizing from Paper Section 5:
equal-notional allocation with 2+2 leverage (2x long, 2x short).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import compute_transaction_cost


@dataclass
class Position:
    """A single stock position."""
    ticker: str
    direction: int              # +1 long, -1 short
    entry_date: pd.Timestamp
    entry_price: float
    notional: float             # dollar amount invested
    quantity: float             # shares (can be fractional)


class PortfolioManager:
    """
    Manages positions, equity, and leverage for the backtest.

    Args:
        initial_equity: Starting portfolio value.
        leverage_long: Maximum long leverage (default: 2.0).
        leverage_short: Maximum short leverage (default: 2.0).
        tc_bps: Transaction cost in basis points per side.
    """

    def __init__(
        self,
        initial_equity: float = 1_000_000.0,
        leverage_long: float = 2.0,
        leverage_short: float = 2.0,
        tc_bps: float = 5.0,
    ):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.leverage_long = leverage_long
        self.leverage_short = leverage_short
        self.tc_bps = tc_bps
        self.positions: dict[str, Position] = {}
        # Hedge positions, keyed by "{stock_ticker}::{hedge_ticker}". Each
        # stock trade opens an offsetting position in the hedge instrument
        # (SPY for PCA, sector ETF for ETF model) sized beta * notional.
        self.hedge_positions: dict[str, Position] = {}
        self.cash = initial_equity
        self.total_costs = 0.0

    @property
    def long_exposure(self) -> float:
        return sum(
            p.notional for p in self.positions.values() if p.direction == 1
        )

    @property
    def short_exposure(self) -> float:
        return sum(
            abs(p.notional) for p in self.positions.values() if p.direction == -1
        )

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    @property
    def net_exposure(self) -> float:
        return self.long_exposure - self.short_exposure

    def compute_notional_per_position(self, n_target_positions: int) -> float:
        """
        Compute equal-notional allocation per position.

        Paper: lambda_t = leverage / n_expected_positions
        The amount invested per stock is equity * lambda_t.
        """
        if n_target_positions <= 0:
            return 0.0
        # Average leverage across long and short
        avg_leverage = (self.leverage_long + self.leverage_short) / 2.0
        return self.equity * avg_leverage / n_target_positions

    def open_position(
        self,
        ticker: str,
        direction: int,
        price: float,
        date: pd.Timestamp,
        notional: float,
    ) -> Position | None:
        """
        Open a new position (bang-bang: full allocation at once).

        Returns the Position if opened, None if already exists.
        """
        if ticker in self.positions:
            return None

        if price <= 0 or notional <= 0:
            return None

        # Check leverage limits
        if direction == 1:
            if self.long_exposure + notional > self.equity * self.leverage_long:
                return None
        else:
            if self.short_exposure + notional > self.equity * self.leverage_short:
                return None

        quantity = notional / price
        cost = compute_transaction_cost(notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost

        pos = Position(
            ticker=ticker,
            direction=direction,
            entry_date=date,
            entry_price=price,
            notional=notional,
            quantity=quantity,
        )
        self.positions[ticker] = pos
        return pos

    def close_position(
        self, ticker: str, price: float, date: pd.Timestamp
    ) -> float:
        """
        Close an existing position. Returns realized PnL.
        """
        if ticker not in self.positions:
            return 0.0

        pos = self.positions[ticker]
        pnl = pos.direction * pos.quantity * (price - pos.entry_price)

        close_notional = abs(pos.quantity * price)
        cost = compute_transaction_cost(close_notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost

        self.cash += pnl
        self.equity += pnl - cost

        del self.positions[ticker]
        return pnl

    def open_hedge_slice(
        self,
        stock_ticker: str,
        hedge_ticker: str,
        stock_direction: int,
        stock_notional: float,
        beta: float,
        hedge_price: float,
        date: pd.Timestamp,
    ) -> Position | None:
        """
        Open an offsetting hedge position sized to cancel the stock trade's
        factor exposure. Signed notional = -stock_direction * beta * notional.
        """
        if (
            hedge_ticker in (None, "none", "")
            or hedge_price <= 0
            or not np.isfinite(beta)
        ):
            return None
        signed = -stock_direction * beta * stock_notional
        hedge_notional = abs(signed)
        if hedge_notional <= 0:
            return None
        hedge_direction = 1 if signed > 0 else -1

        key = f"{stock_ticker}::{hedge_ticker}"
        if key in self.hedge_positions:
            return None

        qty = hedge_notional / hedge_price
        cost = compute_transaction_cost(hedge_notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost

        pos = Position(
            ticker=hedge_ticker,
            direction=hedge_direction,
            entry_date=date,
            entry_price=hedge_price,
            notional=hedge_notional,
            quantity=qty,
        )
        self.hedge_positions[key] = pos
        return pos

    def close_hedge_slice(
        self,
        stock_ticker: str,
        hedge_ticker: str,
        hedge_price: float,
        date: pd.Timestamp,
    ) -> float:
        key = f"{stock_ticker}::{hedge_ticker}"
        if key not in self.hedge_positions:
            return 0.0
        pos = self.hedge_positions[key]
        pnl = pos.direction * pos.quantity * (hedge_price - pos.entry_price)
        close_notional = abs(pos.quantity * hedge_price)
        cost = compute_transaction_cost(close_notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost
        self.cash += pnl
        del self.hedge_positions[key]
        return pnl

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """
        Update equity with unrealized PnL from stock and hedge positions.
        Returns total daily unrealized PnL on stock leg (for diagnostics).
        """
        daily_pnl = 0.0
        for ticker, pos in self.positions.items():
            if ticker in prices and np.isfinite(prices[ticker]):
                daily_pnl += pos.direction * pos.quantity * (
                    prices[ticker] - pos.entry_price
                )

        unrealized_stocks = sum(
            pos.direction * pos.quantity * (
                prices.get(pos.ticker, pos.entry_price) - pos.entry_price
            )
            for pos in self.positions.values()
        )
        unrealized_hedges = sum(
            pos.direction * pos.quantity * (
                prices.get(pos.ticker, pos.entry_price) - pos.entry_price
            )
            for pos in self.hedge_positions.values()
        )
        self.equity = self.cash + unrealized_stocks + unrealized_hedges
        return daily_pnl
