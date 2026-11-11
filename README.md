# Cost-Robust Market-Neutral Momentum in Cryptocurrency Markets

A weekly cross-sectional momentum strategy on a small crypto universe, built to be
**market-neutral** and tested for whether the edge survives **transaction costs**,
**multiple-comparison correction**, and **out-of-sample** data — rather than reported
at its most flattering gross number.

The headline finding is deliberately modest and defensible: momentum generates a
market-neutral return that beats buy-and-hold after realistic costs, and a more
exciting "momentum inverts on the liquidity spectrum" hypothesis was **tested and
rejected**, not asserted.

---

## Strategy in one paragraph

Each week, rank the coins in the universe by recent return (2-week formation), go long
the winners and short the losers in equal dollar size (dollar-neutral), hold for one
week, then rebalance. Because the book is long and short in equal size, the bet is
*winners beat losers*, not *crypto goes up* — so the return should be largely
independent of overall market direction.

---

## Headline results

Universe of 7 coins (BTC, ETH, ADA, DOT, BNB, XRP, MATIC), weekly, 2022–2024
(~156 weeks after aligning histories).

| Metric | Value |
|---|---|
| Net Sharpe (after 15 bps/turnover) | **~1.23** |
| Gross Sharpe (no costs) | ~1.52 |
| Equal-weight buy-and-hold Sharpe | ~0.46 |
| Market beta (vs equal-weight benchmark) | **~0.09** |
| Annualized alpha | ~+25.5% (p ≈ 0.05) |
| Avg weekly turnover | ~0.86x |

*All Sharpes are net of transaction costs unless labeled gross.*

**Reading it:** the strategy roughly triples the risk-adjusted return of buying and
holding the same coins, and its near-zero beta means that return is a genuine
cross-sectional signal rather than disguised market exposure.

---

## The liquidity hypothesis — tested and rejected

An early version of this analysis claimed momentum was stronger in less-liquid coins
(a "liquidity inversion"). When tested properly, that claim did not hold:

- Sorting coins by **realized median dollar volume** into High/Mid/Low liquidity
  buckets, the difference between Low- and High-liquidity momentum returns was
  **statistically indistinguishable from zero (p ≈ 0.99)**.
- No individual liquidity bucket's returns survived **Bonferroni/BH correction**
  against zero.
- The one significant difference in the data — a name-based "Tokens minus Layer1"
  spread (p ≈ 0.009) — reflects **basket composition**, not liquidity: the token
  basket (BNB, XRP, MATIC) contains some of the most liquid assets in crypto, so the
  gap can't be a liquidity-microstructure effect.

The defensible conclusion is narrower and honest: cross-sectional momentum **diverged
sharply between baskets**, but that divergence **does not track liquidity**.

---

## Robustness checks

- **Cost sensitivity** — net Sharpe decays gracefully and stays positive well past a
  realistic cost assumption:

  | bps/turnover | 0 | 5 | 10 | 15 | 25 | 50 |
  |---|---|---|---|---|---|---|
  | Sharpe | 1.52 | 1.42 | 1.33 | 1.23 | 1.03 | 0.54 |

- **Formation-window sweep** — positive net Sharpe across all 1–8 week formation
  windows (strongest at 2 weeks; that specific window is the best case, not a plateau).
- **Alpha/beta decomposition** — one-factor OLS of strategy returns on an equal-weight
  market benchmark (plain NumPy, no `statsmodels` dependency).
- **Out-of-sample split** — 60% in-sample / 40% out-of-sample; signs stayed stable
  (overall IS +0.79 → OOS +1.90).
- **Multiple-comparison correction** — Bonferroni and Benjamini-Hochberg applied
  across the per-bucket tests.
- **Diversification check** — the pooled 7-coin book beat every single-theme sleeve
  (0 of 3), supporting broad exposure over concentration.

---

## Honest caveats

These are stated plainly because the analysis is only as credible as its limitations:

- **Short sample, one regime.** ~156 weeks (2022–2024), a single market cycle, one
  realized path.
- **Survivorship bias.** The 7 coins are ones that survived to the present; this
  flatters results.
- **Out-of-sample beat in-sample.** OOS Sharpe exceeding IS almost everywhere most
  likely reflects a favorable late-sample regime, not a strategy that improves over
  time. Walk-forward validation across multiple regimes is the right next step before
  trusting the point estimates.
- **Marginal alpha significance.** Alpha is significant at roughly p ≈ 0.05, not
  overwhelmingly so.
- **Simplified cost model.** A flat per-turnover cost; it does not scale with order
  size and does not include short funding/borrow costs, so the true net number is
  likely a touch below the reported one.
- **Thin cross-section in 2-coin buckets.** With only two assets, the long/short
  collapses to a near-coin-flip and is very noisy.

---

## Repository structure

```
crypto-momentum-analysis/
├── crypto_momentum_analysis.ipynb   # full analysis: strategy, stats, robustness
├── requirements.txt
├── .gitignore
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook crypto_momentum_analysis.ipynb
```

The notebook pulls daily OHLCV from Binance via `python-binance`
(`Client(tld='US')`). Data access requires network connectivity to Binance; no API
key is needed for the public historical-klines endpoint used here. Run the cells top
to bottom — the analysis is organized into a shared engine (cost-aware momentum +
metrics) followed by benchmark, liquidity-sorted buckets, difference tests, alpha/beta
decomposition, and robustness checks.

---

## Disclaimer

This is a research and backtesting exercise, not investment advice. Backtested results
are not a track record and do not guarantee future performance. Nothing here is a
recommendation to trade any asset.
