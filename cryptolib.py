"""
cryptolib.py - shared data, backtest and performance utilities for the
crypto momentum / reversal notebooks.

Conventions used everywhere
---------------------------
* Data: Binance public daily klines (close and quote-currency volume, USDT ~ USD).
  No API key is needed. api.binance.us is tried first (US users), then api.binance.com.
* Timing: a weight decided at the close of period t is earned over period t+1.
  Every signal uses only data available at its own close (no look-ahead).
* Costs: COST_BPS per unit of one-way turnover = 7 bps commission + 13 bps slippage
  = 20 bps for market orders, as specified in the course brief.
* Statistics: Newey-West (HAC) standard errors for regressions and means, because
  overlapping and autocorrelated returns make naive standard errors too small.
* Performance: risk-free rate = 0. Annualization uses 52 periods (weekly) or
  365 (daily, crypto trades every day).
"""
import time

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats

COST_BPS = 20.0
COST_RATE = COST_BPS / 1e4
PPY_WEEKLY = 52
PPY_DAILY = 365

BINANCE_HOSTS = ["https://api.binance.us", "https://api.binance.com"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "num_trades",
              "taker_base_volume", "taker_quote_volume", "ignore"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_klines(symbol, interval="1d", start="2020-01-01", end=None):
    """Download every kline for `symbol` between start and end (UTC).

    Returns a DataFrame indexed by candle open time with columns close and quote_volume.
    Raises RuntimeError if no host returns data.
    """
    end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            rows, cur = [], start_ms
            while cur < end_ms:
                resp = requests.get(
                    f"{host}/api/v3/klines",
                    params=dict(symbol=symbol, interval=interval,
                                startTime=cur, endTime=end_ms, limit=1000),
                    timeout=30)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                cur = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
                time.sleep(0.05)
            if rows:
                df = pd.DataFrame(rows, columns=KLINE_COLS)
                df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms")
                return df[["close", "quote_volume"]].astype(float)
        except Exception as err:  # fall through to the next host
            last_err = err
    raise RuntimeError(f"no data for {symbol}: {last_err}")


def load_universe(symbols, start="2020-01-01", end=None):
    """Download daily data for a list of tickers such as 'BTCUSDT'.

    Returns (close, quote_volume), each a DataFrame with coin names as columns
    (e.g. 'BTC') on a complete daily index. Missing coins are reported and skipped.
    """
    closes, qvols = {}, {}
    for sym in symbols:
        coin = sym.replace("USDT", "")
        try:
            d = fetch_klines(sym, "1d", start, end)
        except RuntimeError as err:
            print(f"  SKIPPED {sym}: {err}")
            continue
        closes[coin] = d["close"]
        qvols[coin] = d["quote_volume"]
        print(f"  {sym:<10} {d.index[0].date()} -> {d.index[-1].date()}  ({len(d)} days)")
    close = pd.DataFrame(closes).sort_index()
    qvol = pd.DataFrame(qvols).sort_index()
    full = pd.date_range(close.index.min(), close.index.max(), freq="D")
    return close.reindex(full), qvol.reindex(full)


def coverage_table(close):
    """First date, last date and number of observations for every coin."""
    rows = []
    for c in close.columns:
        s = close[c].dropna()
        rows.append({"coin": c, "first": s.index[0].date(),
                     "last": s.index[-1].date(), "obs": len(s)})
    return pd.DataFrame(rows).set_index("coin")


def to_weekly(close):
    """Friday close of each week."""
    return close.resample("W-FRI").last()


def common_window(df):
    """Keep only rows where every column has a value (no partial histories)."""
    return df.dropna(how="any")


# ---------------------------------------------------------------------------
# Portfolio construction and backtesting
# ---------------------------------------------------------------------------
def rank_weights(signal):
    """Cross-sectional rank-demeaned, normalized weights (Homework 8 / Dobrynskaya style).

    Each row: ranks minus their mean (so weights sum to zero, dollar-neutral in rank
    space), then divided by the sum of absolute values (so gross exposure = 1).
    Rows with no signal get zero weight.
    """
    r = signal.rank(axis=1)
    r = r.sub(r.mean(axis=1), axis=0)
    denom = r.abs().sum(axis=1).replace(0, np.nan)
    return r.div(denom, axis=0).fillna(0.0)


def backtest(weights, ret, cost_rate=COST_RATE):
    """Earn the weights decided at t-1 over the return at t, net of turnover costs.

    Returns a DataFrame with columns gross, turnover, cost, net.
    Turnover = sum of absolute changes in held weights (one-way).
    """
    held = weights.shift(1).fillna(0.0)
    gross = (held * ret).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    cost = cost_rate * turnover
    return pd.DataFrame({"gross": gross, "turnover": turnover,
                         "cost": cost, "net": gross - cost})


def momentum_backtest(ret, formation=2, reverse=False, cost_rate=COST_RATE):
    """Cross-sectional momentum (or reversal) on trailing average returns.

    formation: number of periods in the trailing average.
    reverse=True flips the sign of the signal (short-term reversal).
    """
    sig = ret.rolling(formation, min_periods=formation).mean()
    if reverse:
        sig = -sig
    return backtest(rank_weights(sig), ret, cost_rate)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def perf_metrics(r, ppy=PPY_WEEKLY):
    """Return, volatility, Sharpe, drawdown and significance for one return series."""
    r = pd.Series(r).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    ann_vol = sd * np.sqrt(ppy)
    eq = (1 + r).cumprod()
    years = n / ppy
    final = eq.iloc[-1]
    cagr = final ** (1 / years) - 1 if final > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    downside = r[r < 0].std(ddof=1) * np.sqrt(ppy)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return pd.Series({
        "Total return": final - 1,
        "CAGR": cagr,
        "Ann. return (mean x ppy)": mu * ppy,
        "Ann. volatility": ann_vol,
        "Sharpe": mu * ppy / ann_vol if ann_vol > 0 else np.nan,
        "Sortino": mu * ppy / downside if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if (mdd < 0 and not np.isnan(cagr)) else np.nan,
        "Win rate": (r > 0).mean(),
        "t-stat (mean, iid)": t,
        "p-value (mean, iid)": 2 * (1 - stats.t.cdf(abs(t), n - 1)) if not np.isnan(t) else np.nan,
        "Periods": n,
    })


def metrics_table(series_dict, ppy=PPY_WEEKLY):
    """One column per strategy, one row per metric."""
    return pd.DataFrame({name: perf_metrics(r, ppy) for name, r in series_dict.items()})


def cost_summary(bt, ppy=PPY_WEEKLY):
    """Annual turnover, annual cost drag and the breakeven all-in cost in bps."""
    to = bt["turnover"].mean()
    return pd.Series({
        "Ann. one-way turnover": to * ppy,
        "Ann. cost drag": bt["cost"].mean() * ppy,
        "Breakeven cost (bps per unit turnover)": (bt["gross"].mean() / to * 1e4) if to > 0 else np.nan,
    })


def net_sharpe_at_cost(bt, bps, ppy=PPY_WEEKLY):
    """Sharpe of the strategy if the all-in cost were `bps` per unit of turnover."""
    r = bt["gross"] - bps / 1e4 * bt["turnover"]
    sd = r.std(ddof=1)
    return r.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan


def yearly_table(r, ppy=PPY_WEEKLY):
    """Return, volatility, Sharpe and sample size by calendar year."""
    r = pd.Series(r).dropna()
    g = r.groupby(r.index.year)
    vol = g.std(ddof=1) * np.sqrt(ppy)
    return pd.DataFrame({"Ann. return": g.mean() * ppy, "Ann. vol": vol,
                         "Sharpe": g.mean() * ppy / vol, "Periods": g.size()})


# ---------------------------------------------------------------------------
# Regressions and tests
# ---------------------------------------------------------------------------
def alpha_beta(strat, bench, ppy=PPY_WEEKLY, maxlags=4):
    """OLS: strategy = alpha + beta * benchmark, Newey-West (HAC) standard errors."""
    d = pd.concat([pd.Series(strat).rename("strat"),
                   pd.Series(bench).rename("bench")], axis=1).dropna()
    X = sm.add_constant(d["bench"])
    m = sm.OLS(d["strat"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return pd.Series({
        "Alpha (ann.)": m.params["const"] * ppy,
        "Alpha t-stat (HAC)": m.tvalues["const"],
        "Alpha p-value": m.pvalues["const"],
        "Beta": m.params["bench"],
        "Beta t-stat (HAC)": m.tvalues["bench"],
        "Beta p-value": m.pvalues["bench"],
        "R-squared": m.rsquared,
        "Observations": int(m.nobs),
    })


def mean_test_hac(x, maxlags=4):
    """Test whether the mean of x is zero, with Newey-West standard errors."""
    x = pd.Series(x).dropna()
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"mean": m.params[0], "t-stat": m.tvalues[0],
            "p-value": m.pvalues[0], "n": len(x)}


def mean_diff_test(a, b, maxlags=4):
    """Test whether mean(a - b) is zero, with Newey-West standard errors."""
    d = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    res = mean_test_hac(d["a"] - d["b"], maxlags)
    res["mean diff"] = res.pop("mean")
    return res


def bootstrap_sharpe(r, ppy=PPY_WEEKLY, n_boot=2000, block=4, seed=0):
    """Moving-block bootstrap of the Sharpe ratio.

    Resamples overlapping blocks of `block` consecutive periods so that
    autocorrelation is preserved. Returns (95% CI as array, all bootstrap Sharpes).
    """
    x = pd.Series(r).dropna().values
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        samp = np.concatenate([x[s:s + block] for s in starts])[:n]
        sd = samp.std(ddof=1)
        sharpes[b] = samp.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan
    ci = np.nanpercentile(sharpes, [2.5, 97.5])
    return ci, sharpes
