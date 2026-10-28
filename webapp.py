"""Streamlit demo: pick a ticker, train MSE vs Profit-Aware GRUs live, and
inspect the resulting trade log side by side.

Reuses the exact same data pipeline, model, and training loop as
``main.py --compare --trade-log`` (same defaults as benchmark.py), so the
numbers shown here match what the CLI would produce for identical
arguments. Data is loaded from the CSV caches under data/. Once
benchmark.py has run, every ticker in the thesis universe (the PSEi
constituents and the index, see benchmark.py's ALL_SYMBOLS) has one, so no network
access is needed for a live defense demo.

Run with:

    streamlit run webapp.py

The theme lives in .streamlit/config.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from benchmark import INDEX_SYMBOL, PSEI_STOCKS, buy_and_hold_return, is_degenerate_signal
from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch
from utils.baselines import evaluate_price_strategies
from utils.metrics import running_balance, sharpe_like, trade_log_records
from utils.pipeline import cache_path_for, prepare_data
from utils.preprocessing import load_stock_data
from utils.plotting import format_date
from utils.chart_style import COLOR_MSE, COLOR_PA, COLOR_REF
from utils.trading import PH_BUY_COST, PH_SELL_COST, calibrate_alpha

STRATEGY_LABELS = {"sma_crossover": "SMA crossover", "rsi": "RSI", "macd": "MACD"}
STRATEGY_COLORS = ["#8a5fbf", "#4fa07a", "#c94f6d"]

# Shared thesis palette (utils.chart_style): blue = MSE, orange = profit-aware,
# gray = buy & hold. The three TA baselines are secondary context.

st.set_page_config(page_title="GRU: MSE vs profit-aware", page_icon=":material/query_stats:", layout="wide")


def ticker_catalog() -> list[dict]:
    """Every symbol in the thesis universe: the PSEi index (the primary series,
    selected by default), then the constituent stocks."""
    return [{"symbol": INDEX_SYMBOL, "kind": "PSEi index"}] + [
        {"symbol": s, "kind": "PSEi stock"} for s in PSEI_STOCKS
    ]


def run_side(loss_type: str, data, alpha: float, device: torch.device, cfg: dict) -> dict:
    """Train one loss variant and return its test metrics + trade log."""
    split = data.split
    output_scale = (cfg["output_cap_std"] / alpha) if cfg["output_cap_std"] > 0 else None

    torch.manual_seed(cfg["seed"])
    model = GRUReturnPredictor(
        input_size=split.x_train.shape[-1],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_scale=output_scale,
    ).to(device)

    config = TrainingConfig(
        loss_type=loss_type,
        alpha=alpha,
        loss_lambda=cfg["loss_lambda"],
        buy_cost_rate=cfg["buy_cost"],
        sell_cost_rate=cfg["sell_cost"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=64,
        epochs=cfg["epochs"],
    )
    fit(model, data.train_loader, data.val_loader, config, device, capital=cfg["capital"])

    test_metrics = run_epoch(
        model, data.test_loader, None, alpha, cfg["buy_cost"], cfg["sell_cost"], device, loss_type=loss_type,
    )
    test_metrics["degenerate"] = is_degenerate_signal(test_metrics["signal_np"])
    records, summary = trade_log_records(
        test_metrics["signal_np"], test_metrics["actual_np"], test_metrics["pred_np"],
        split.dates_test, cfg["capital"], buy_cost_rate=cfg["buy_cost"], sell_cost_rate=cfg["sell_cost"],
    )
    test_metrics["records"] = records
    test_metrics["summary"] = summary
    return test_metrics


@st.cache_resource(show_spinner=False)
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_comparison(symbol: str, cfg: dict) -> dict:
    device = _device()
    data = prepare_data(symbol, cfg["start"], None, cfg["sequence_length"], 64)
    alpha = cfg["alpha"] if cfg["alpha"] else calibrate_alpha(data.split.y_train)

    mse_metrics = run_side("mse", data, alpha, device, cfg)
    pa_metrics = run_side("profit-aware", data, alpha, device, cfg)

    buy_hold_return = buy_and_hold_return(mse_metrics["actual_np"])
    buy_hold_balance = cfg["capital"] * np.cumprod(1.0 + mse_metrics["actual_np"])
    buy_hold_sharpe = sharpe_like(mse_metrics["actual_np"])

    price_df = load_stock_data(symbol, cfg["start"], None, cache_path_for(symbol, cfg["start"], None))
    strategies = evaluate_price_strategies(
        price_df["Close"], data.split.dates_test, mse_metrics["actual_np"],
        buy_cost_rate=cfg["buy_cost"], sell_cost_rate=cfg["sell_cost"],
    )
    strategy_balances = {
        name: running_balance(m["signal"], mse_metrics["actual_np"], cfg["capital"], cfg["buy_cost"], cfg["sell_cost"])
        for name, m in strategies.items()
    }

    return {
        "symbol": symbol,
        "split": data.split,
        "alpha": alpha,
        "capital": cfg["capital"],
        "mse": mse_metrics,
        "pa": pa_metrics,
        "buy_hold_return": buy_hold_return,
        "buy_hold_balance": buy_hold_balance,
        "buy_hold_sharpe": buy_hold_sharpe,
        "strategies": strategies,
        "strategy_balances": strategy_balances,
    }


def side_switches(signal: np.ndarray) -> int:
    """How many times the executed position changes side (long <-> short)."""
    sides = np.sign(signal)
    sides = sides[sides != 0]
    return int(np.sum(sides[1:] != sides[:-1]))


def metrics_table(result: dict) -> pd.DataFrame:
    """One row per policy on the shared test window."""
    rows = []
    for name, m in (("MSE baseline", result["mse"]), ("Profit-aware", result["pa"])):
        rows.append({
            "Policy": name, "Directional accuracy": m["directional_acc"], "Cumulative return": m["cum_profit"],
            "Geometric return": m["cum_profit_geo"], "Sharpe-like": m["sharpe_like"], "RMSE": m["rmse"],
            "Side switches": side_switches(m["signal_np"]),
        })
    rows.append({"Policy": "Buy & hold", "Geometric return": result["buy_hold_return"],
                 "Sharpe-like": result["buy_hold_sharpe"]})
    for name, m in result["strategies"].items():
        rows.append({
            "Policy": STRATEGY_LABELS[name], "Directional accuracy": m["directional_acc"],
            "Cumulative return": m["cum_profit"], "Geometric return": m["cum_profit_geo"],
            "Sharpe-like": m["sharpe_like"],
        })
    return pd.DataFrame(rows).set_index("Policy")


def render_trade_log(m: dict) -> None:
    summary = m["summary"]
    with st.container(horizontal=True):
        st.metric("Wins / losses", f"{summary['wins']} / {summary['losses']}", border=True)
        st.metric("Best day", f"{summary['best_trade']*100:+.2f}%", summary["best_date"] or None,
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("Worst day", f"{summary['worst_trade']*100:+.2f}%", summary["worst_date"] or None,
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("Final balance", f"{summary['final_geo_balance']:,.0f} PHP", border=True)

    df = pd.DataFrame(m["records"])
    df["side"] = np.where(df["signal"] > 0, "Long", np.where(df["signal"] < 0, "Short", "Flat"))
    df["return"] *= 100
    df.index = range(1, len(df) + 1)
    df = df[["date", "return", "pred", "signal", "side", "trade_pnl", "add_balance", "geo_balance"]]
    st.dataframe(
        df,
        height=420,
        column_config={
            "date": st.column_config.TextColumn("Date"),
            "return": st.column_config.NumberColumn("Actual return", format="%.2f%%"),
            "pred": st.column_config.NumberColumn("Predicted return", format="%.4f"),
            "signal": st.column_config.NumberColumn("Position", format="%.3f"),
            "side": st.column_config.TextColumn("Side"),
            "trade_pnl": st.column_config.NumberColumn("Day P&L (PHP)", format="localized"),
            "add_balance": st.column_config.NumberColumn("Additive balance (PHP)", format="localized"),
            "geo_balance": st.column_config.NumberColumn("Compounded balance (PHP)", format="localized"),
        },
    )


catalog = ticker_catalog()
labels = [f"{c['symbol'].split(':')[-1]}  ({c['kind']})" for c in catalog]

with st.sidebar:
    st.header("Setup")
    idx = st.selectbox("Ticker", options=range(len(catalog)), format_func=lambda i: labels[i])
    symbol = catalog[idx]["symbol"]

    with st.expander("Advanced settings", icon=":material/tune:"):
        sequence_length = st.number_input("Sequence length", 5, 120, 30)
        epochs = st.slider("Epochs", 5, 100, 50, step=5)
        hidden_size = st.number_input("Hidden size", 8, 256, 64, step=8)
        num_layers = st.number_input("GRU layers", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.6, 0.2, step=0.05)
        capital = st.number_input("Capital (PHP)", 1_000.0, 10_000_000.0, 100_000.0, step=10_000.0)
        buy_cost = st.number_input("Buy cost rate", 0.0, 0.05, PH_BUY_COST, step=0.0001, format="%.5f",
                                   help="Commission, VAT and PSE/SCCP fees per unit of position increase")
        sell_cost = st.number_input("Sell cost rate", 0.0, 0.05, PH_SELL_COST, step=0.0001, format="%.5f",
                                    help="Buy-side fees plus the 0.1% stock transaction tax, per unit of position decrease")
        loss_lambda = st.number_input("MSE calibration weight (lambda)", 0.0, 5.0, 0.7, step=0.1)
        seed = st.number_input("Random seed", 0, 10_000, 42)

    run_clicked = st.button("Train both models", type="primary", icon=":material/play_arrow:", width="stretch")
    st.caption("Single chronological 70/15/15 split for a live demo. The thesis results use "
               "4-fold walk-forward evaluation; see `results/`.")

cfg = dict(
    start="2018-01-01",
    sequence_length=int(sequence_length),
    epochs=int(epochs),
    hidden_size=int(hidden_size),
    num_layers=int(num_layers),
    dropout=float(dropout),
    capital=float(capital),
    buy_cost=float(buy_cost),
    sell_cost=float(sell_cost),
    loss_lambda=float(loss_lambda),
    output_cap_std=5.0,
    lr=1e-3,
    weight_decay=1e-3,
    seed=int(seed),
    alpha=None,
)

st.title("GRU return forecasting: MSE vs profit-aware loss")
st.caption("Both models share the data, architecture, seed and chronological split. "
           "Only the training loss differs.")

if run_clicked:
    with st.spinner(f"Training both GRUs on {symbol} ({epochs} epochs each)..."):
        t0 = time.time()
        st.session_state["result"] = run_comparison(symbol, cfg)
        st.session_state["elapsed"] = time.time() - t0
        st.session_state["cfg"] = cfg

result = st.session_state.get("result")
if result is None:
    st.info("Choose a ticker in the sidebar, then select **Train both models**.", icon=":material/info:")
    st.stop()

run_cfg = st.session_state.get("cfg", cfg)
split = result["split"]
mse, pa = result["mse"], result["pa"]
capital = result["capital"]

st.subheader(result["symbol"])
st.markdown(
    f":gray-badge[:material/date_range: {format_date(split.dates_test[0])} to {format_date(split.dates_test[-1])}] "
    f":gray-badge[{len(split.dates_test)} test days] "
    f":gray-badge[alpha {result['alpha']:.2f}] "
    f":gray-badge[lambda {run_cfg['loss_lambda']:g}] "
    f":gray-badge[costs {run_cfg['buy_cost']:.3%} buy, {run_cfg['sell_cost']:.3%} sell] "
    f":gray-badge[:material/timer: {st.session_state.get('elapsed', 0):.0f}s on {_device()}]"
)

with st.container(horizontal=True):
    st.metric("Profit-aware return", f"{pa['cum_profit_geo']:+.2%}",
              f"{(pa['cum_profit_geo'] - mse['cum_profit_geo']) * 100:+.2f} pp vs MSE", border=True)
    st.metric("MSE baseline return", f"{mse['cum_profit_geo']:+.2%}",
              f"{mse['cum_profit_geo'] * capital:+,.0f} PHP", border=True)
    st.metric("Buy & hold return", f"{result['buy_hold_return']:+.2%}", "always long, no costs",
              delta_color="off", delta_arrow="off", border=True)
    st.metric("Profit-aware directional accuracy", f"{pa['directional_acc']:.1%}",
              f"{(pa['directional_acc'] - mse['directional_acc']) * 100:+.1f} pp vs MSE", border=True)
st.caption("Returns are compounded over the test window, after trading costs.")

one_sided = [name for name, m in (("the MSE baseline", mse), ("the profit-aware model", pa)) if m.get("degenerate")]
if one_sided:
    subject = " and ".join(one_sided)
    verb = "never changes" if len(one_sided) == 1 else "never change"
    st.warning(
        f"{subject[0].upper()}{subject[1:]} {verb} side over the test window, so the return reflects the "
        "market's drift in that direction rather than day-to-day forecasting.", icon=":material/warning:",
    )

tab_balance, tab_metrics, tab_log = st.tabs([
    ":material/show_chart: Balance", ":material/table_chart: All metrics", ":material/receipt_long: Trade log",
])

with tab_balance:
    dates = [format_date(d) for d in split.dates_test]
    chart_df = pd.DataFrame(
        {
            "MSE baseline": [r["geo_balance"] for r in mse["records"]],
            "Profit-aware": [r["geo_balance"] for r in pa["records"]],
            "Buy & hold": result["buy_hold_balance"],
            **{STRATEGY_LABELS[name]: bal for name, bal in result["strategy_balances"].items()},
        },
        index=pd.Index(dates, name="Date"),
    )
    st.line_chart(chart_df, color=[COLOR_MSE, COLOR_PA, COLOR_REF, *STRATEGY_COLORS], y_label="Balance (PHP)")
    st.caption(f"Growth of {capital:,.0f} PHP, compounded daily after costs. "
               "Buy & hold is shown without costs, as context.")

with tab_metrics:
    st.dataframe(
        metrics_table(result),
        column_config={
            "Directional accuracy": st.column_config.NumberColumn(format="percent"),
            "Cumulative return": st.column_config.NumberColumn(format="percent"),
            "Geometric return": st.column_config.NumberColumn(format="percent"),
            "Sharpe-like": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.5f"),
            "Side switches": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption("Cumulative return adds the daily net returns; geometric return compounds them. "
               "SMA crossover, RSI and MACD trade the same window under the same costs.")

with tab_log:
    which = st.segmented_control("Model", ["Profit-aware", "MSE baseline"], default="Profit-aware",
                                 label_visibility="collapsed")
    render_trade_log(mse if which == "MSE baseline" else pa)
