"""Cross-asset features: benchmark index returns, relative strength, rolling beta and residual returns.

Rolling estimates at bar T are fitted on the window ending at T (inclusive), so they describe the
relationship up to now and never use later bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA_WINDOW = 60
LASSO_ALPHA = 3e-8  # L1 penalty on the slope; depends on the return scale of the instruments


def index_features(index_bars: pd.DataFrame, name: str) -> pd.DataFrame:
    """OHLC and close-to-close return of a benchmark index, with columns prefixed by `name`."""
    out = index_bars[["open", "high", "low", "close"]].add_prefix(f"{name}_")
    out[f"{name}_return"] = index_bars["close"].pct_change()
    return out


def rolling_lasso_beta(y: pd.Series, x: pd.Series, window: int = BETA_WINDOW, alpha: float = LASSO_ALPHA) -> pd.Series:
    """Rolling LASSO slope of y on x (with intercept), in closed form.

    With a single regressor, scikit-learn's Lasso objective (1/2n)||y - a - bx||^2 + alpha|b| is
    solved by soft-thresholding the covariance: b = sign(c) * max(|c| - alpha, 0) / var(x), where c
    and var(x) are population moments of the window. Windows with a missing value give NaN.
    """
    full = (x.notna() & y.notna()).rolling(window).sum() == window
    cov = y.rolling(window).cov(x, ddof=0)
    var = x.rolling(window).var(ddof=0)
    beta = np.sign(cov) * (cov.abs() - alpha).clip(lower=0) / var.where(var > 1e-16)
    return beta.where(full)


def relative_features(
    stock_close: pd.Series,
    stock_vwap: pd.Series,
    index_close: pd.Series,
    window: int = BETA_WINDOW,
    alpha: float = LASSO_ALPHA,
) -> pd.DataFrame:
    """Relative strength, rolling beta / correlation and beta-adjusted residual returns."""
    out = pd.DataFrame(index=stock_close.index)
    r = stock_close.pct_change()
    m = index_close.pct_change()
    out["stock_return"] = r
    out["return_rs"] = r - m

    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()
    beta = cov / var.where(var > 1e-16)
    rho = r.rolling(window).corr(m)
    out["beta_ols"] = beta
    out["corr"] = rho
    out["return_rs_ols"] = r - beta * m
    out["return_rs_ols_r2"] = r - beta * m * rho**2

    beta_lasso = rolling_lasso_beta(r, m, window, alpha)
    out["beta_lasso"] = beta_lasso
    out["return_rs_lasso"] = r - beta_lasso * m

    # Return measured at the bar's average execution price rather than its last print.
    out["return_rs_vwap"] = stock_vwap.pct_change() - m
    return out
