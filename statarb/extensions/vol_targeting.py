"""
Volatility-Targeted Position Sizing.

Replaces equal-notional allocation with vol-parity sizing so each position
contributes roughly equal risk. The risk of position i is:

    risk_i = notional_i * sigma_eq_i

Setting risk_i = constant and using cross-sectional median sigma_eq as the
target gives:

    notional_i = base_notional * (target_sigma / sigma_eq_i)

Stocks with high equilibrium vol receive smaller notional; stable stocks
receive more. A floor and cap prevent degenerate positions.
"""
import numpy as np

from statarb.signals.ou_estimator import OUParams


class VolTargetedSizer:
    """
    Scale position notionals inversely with residual equilibrium volatility.

    Args:
        floor_multiplier: Minimum scale factor relative to equal-notional
            (prevents very small positions for high-vol stocks).
        cap_multiplier: Maximum scale factor relative to equal-notional
            (prevents very large positions for unusually calm stocks).
    """

    def __init__(
        self,
        floor_multiplier: float = 0.2,
        cap_multiplier: float = 5.0,
    ):
        self.floor_multiplier = floor_multiplier
        self.cap_multiplier = cap_multiplier

    def compute_target_sigma(self, ou_params: dict[str, OUParams]) -> float:
        """
        Compute the cross-sectional median sigma_eq across eligible stocks.

        This is used as the normalization target so that the median stock
        receives exactly the equal-notional allocation, preserving average
        leverage.

        Args:
            ou_params: Dict mapping ticker -> OUParams for eligible stocks.

        Returns:
            Median sigma_eq, or 1.0 if no valid params.
        """
        sigmas = [p.sigma_eq for p in ou_params.values() if p.sigma_eq > 0]
        if not sigmas:
            return 1.0
        return float(np.median(sigmas))

    def scale_notional(
        self,
        base_notional: float,
        sigma_eq: float,
        target_sigma: float,
    ) -> float:
        """
        Scale base_notional so position risk equals target risk level.

        Args:
            base_notional: Equal-notional baseline for this position.
            sigma_eq: Equilibrium volatility of this stock's residual.
            target_sigma: Cross-sectional target (typically median sigma_eq).

        Returns:
            Scaled dollar notional for the position.
        """
        if sigma_eq <= 0 or target_sigma <= 0:
            return base_notional

        scale = target_sigma / sigma_eq
        scale = float(np.clip(scale, self.floor_multiplier, self.cap_multiplier))
        return base_notional * scale
