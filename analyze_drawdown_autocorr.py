"""Max drawdown and lag-1 autocorrelation over the exact GRU benchmark test window.

Reuses benchmark.py's own symbol lists and utils/pipeline.py's data-loading +
windowing + train_val_test_split pipeline, so the test-window dates here are
guaranteed identical to whatever benchmark.py produced for the same
--start/--end/--sequence-length (defaults: 2018-01-01 / latest / 30). Never
refetches: every symbol must already have a CSV cache under data/, written by
utils.preprocessing.load_stock_data via benchmark.py or main.py. Missing
caches are reported, not silently skipped or downloaded.

Usage:
    python analyze_drawdown_autocorr.py
    python analyze_drawdown_autocorr.py --market ph
    python analyze_drawdown_autocorr.py --start 2018-01-01 --sequence-length 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import MARKETS
from utils.pipeline import build_sequences, cache_path_for
from utils.preprocessing import train_val_test_split


def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of the cumulative-return series built from `returns`.

    Cumulative price starts at 1.0 within the window itself (no external
    anchor before the window), tracks the running peak, and reports the most
    negative (peak - trough) / peak seen -- as a fraction, e.g. -0.308.
    """
    cum = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(cum)
    drawdown = (cum - running_peak) / running_peak
    return float(drawdown.min())


def lag1_autocorr(returns: np.ndarray) -> float:
    return float(pd.Series(returns).autocorr(lag=1))


def autocorr_significant(ac: float, n: int) -> bool:
    """True if |ac| exceeds the 95% white-noise band ±1.96/sqrt(n) (Bartlett's formula)."""
    return abs(ac) > 1.96 / np.sqrt(n)


def autocorr_interpretation(mean_ac: float) -> str:
    """Plain-language read of a market's mean lag-1 autocorrelation."""
    if abs(mean_ac) < 0.02:
        return "a day tells you almost nothing about tomorrow"
    if mean_ac < 0:
        return "up days tend to be followed by down days"
    return "up days tend to be followed by up days"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Max drawdown + lag-1 autocorrelation over the GRU benchmark's test split"
    )
    parser.add_argument("--market", type=str, default="all", choices=["us", "ph", "all"])
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--output", type=str, default="drawdown_autocorr_summary.csv")
    args = parser.parse_args()

    if args.market == "all":
        symbol_markets = [("PH" if ":" in s else "US", s) for m in MARKETS.values() for tier in m.values() for s in tier]
    else:
        market_tag = args.market.upper()
        symbol_markets = [(market_tag, s) for tier in MARKETS[args.market].values() for s in tier]

    rows: list[dict] = []
    missing: list[str] = []

    for market, symbol in symbol_markets:
        cache_path = cache_path_for(symbol, args.start, args.end)
        if not Path(cache_path).exists():
            missing.append(f"{symbol} (expected {cache_path})")
            continue

        x, y, seq_dates, _dates, _returns = build_sequences(
            symbol, args.start, args.end, args.sequence_length
        )
        split = train_val_test_split(x, y, dates=seq_dates)

        if len(split.y_test) == 0:
            missing.append(f"{symbol} (cached, but test split is empty)")
            continue

        dd = max_drawdown(split.y_test)
        ac = lag1_autocorr(split.y_test)
        n_test = len(split.y_test)
        rows.append(
            {
                "symbol": symbol,
                "market": market,
                "max_drawdown": dd,
                "lag1_autocorrelation": ac,
                "significant_95": autocorr_significant(ac, n_test),
                "test_start": pd.Timestamp(split.dates_test[0]).date().isoformat(),
                "test_end": pd.Timestamp(split.dates_test[-1]).date().isoformat(),
                "n_test": n_test,
            }
        )

    if missing:
        print("=" * 78)
        print(f"  MISSING / INCOMPLETE CACHE ({len(missing)} symbol(s)) -- not computed, not guessed:")
        print("=" * 78)
        for m in missing:
            print(f"  - {m}")
        print()

    if not rows:
        print("No symbols had usable cached data. Nothing to report.")
        return

    df = pd.DataFrame(rows)

    print("=" * 96)
    print("  PER-SYMBOL: MAX DRAWDOWN & LAG-1 AUTOCORRELATION (test split only)")
    print("=" * 96)
    print(
        f"  {'Symbol':<12}{'Market':<8}{'MaxDD':>10}{'LagAC':>10}"
        f"  {'TestStart':>12}  {'TestEnd':>12}{'N':>6}"
    )
    print(f"  {'-'*90}")
    for r in rows:
        print(
            f"  {r['symbol']:<12}{r['market']:<8}{r['max_drawdown']*100:>9.1f}%{r['lag1_autocorrelation']:>10.4f}"
            f"  {r['test_start']:>12}  {r['test_end']:>12}{r['n_test']:>6}"
        )

    print(f"\n{'='*78}")
    print("  GROUP SUMMARY (mean / median across symbols)")
    print(f"{'='*78}")
    print(f"  {'Market':<8}{'MaxDD mean':>14}{'MaxDD median':>16}{'LagAC mean':>14}{'LagAC median':>16}{'Sig@95%':>12}")
    for market in sorted(df["market"].unique()):
        sub = df[df["market"] == market]
        n_sig = int(sub["significant_95"].sum())
        n_total = len(sub)
        print(
            f"  {market:<8}"
            f"{sub['max_drawdown'].mean()*100:>13.1f}%"
            f"{sub['max_drawdown'].median()*100:>15.1f}%"
            f"{sub['lag1_autocorrelation'].mean():>14.4f}"
            f"{sub['lag1_autocorrelation'].median():>16.4f}"
            f"{f'{n_sig}/{n_total}':>12}"
        )

    print(
        "\n  Sig@95% = symbols whose |lag-1 autocorrelation| exceeds the white-noise band "
        "+/-1.96/sqrt(n) (Bartlett's formula) -- i.e. a real (non-random) day-to-day signal,"
        "\n  not just noise. A negative mean AC means up days tend to be followed by down days;"
        " an AC near zero means a day tells you almost nothing about tomorrow."
    )

    print(f"\n{'='*78}")
    print("  TEST-WINDOW DATE RANGE PER SYMBOL")
    print(f"{'='*78}")
    for r in rows:
        print(f"  {r['symbol']:<12} {r['test_start']} -> {r['test_end']}  ({r['n_test']} days)")

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-symbol results to {out_path}")

    markets = sorted(df["market"].unique())
    comparison_path = out_path.parent / "market_comparison.md"
    lines = [
        "| Metric | " + " | ".join(f"{m} Market" for m in markets) + " |",
        "| --- | " + " | ".join("---" for _ in markets) + " |",
    ]
    mean_ac_cells = []
    sig_cells = []
    for m in markets:
        sub = df[df["market"] == m]
        mean_ac = sub["lag1_autocorrelation"].mean()
        n_sig = int(sub["significant_95"].sum())
        n_total = len(sub)
        mean_ac_cells.append(f"{mean_ac:+.3f} ({autocorr_interpretation(mean_ac)})")
        sig_cells.append(f"{n_sig}/{n_total}")
    lines.append("| Mean lag-1 autocorrelation | " + " | ".join(mean_ac_cells) + " |")
    lines.append("| Symbols with a real signal (95%) | " + " | ".join(sig_cells) + " |")

    print(f"\n{'='*78}")
    print("  MARKET COMPARISON")
    print(f"{'='*78}")
    for line in lines:
        print("  " + line)

    comparison_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved market comparison to {comparison_path}")


if __name__ == "__main__":
    main()
