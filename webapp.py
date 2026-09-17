"""Streamlit demo: pick a ticker, train MSE vs Profit-Aware GRUs live, and
inspect the resulting trade log side by side.

Reuses the exact same data pipeline, model, and training loop as
``main.py --compare --trade-log`` (same defaults as benchmark.py), so the
numbers shown here match what the CLI would produce for identical
arguments. Data is loaded from the CSV caches under data/ -- every ticker
in the thesis benchmark universe (see benchmark.py's MARKETS) already has
one, so no network access is needed for a live defense demo.

Run with:

    streamlit run webapp.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from benchmark import MARKETS, buy_and_hold_return, is_degenerate_signal
from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch
from utils.metrics import trade_log_records
from utils.pipeline import prepare_data
from utils.plotting import format_date
from utils.trading import calibrate_alpha

# Palette (validated categorical slots 1/2 + muted reference gray -- see
# dataviz skill): blue = MSE, orange = Profit-Aware, gray = Buy & Hold.
COLOR_MSE = "#2a78d6"
COLOR_PA = "#eb6834"
COLOR_REF = "#898781"
INK = "#0b0b0b"
GRID = "#e1e0d9"

st.set_page_config(page_title="GRU: MSE vs Profit-Aware", page_icon="📈", layout="wide")


def ticker_catalog() -> list[dict]:
    """Every (symbol, market, tier) in the thesis benchmark universe."""
    catalog = []
    for market_name, tiers in MARKETS.items():
        for tier_name, symbols in tiers.items():
            for symbol in symbols:
                catalog.append({"symbol": symbol, "market": market_name.upper(), "tier": tier_name})
    return sorted(catalog, key=lambda c: (c["market"], c["tier"], c["symbol"]))


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
        transaction_cost_rate=cfg["transaction_cost"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=64,
        epochs=cfg["epochs"],
    )
    fit(model, data.train_loader, data.val_loader, config, device, capital=cfg["capital"])

    test_metrics = run_epoch(
        model, data.test_loader, None, alpha, cfg["transaction_cost"], device, loss_type=loss_type,
    )
    test_metrics["degenerate"] = is_degenerate_signal(test_metrics["signal_np"])
    records, summary = trade_log_records(
        test_metrics["signal_np"], test_metrics["actual_np"], test_metrics["pred_np"],
        split.dates_test, cfg["capital"], transaction_cost_rate=cfg["transaction_cost"],
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

    return {
        "symbol": symbol,
        "split": data.split,
        "alpha": alpha,
        "capital": cfg["capital"],
        "mse": mse_metrics,
        "pa": pa_metrics,
        "buy_hold_return": buy_hold_return,
        "buy_hold_balance": buy_hold_balance,
    }


def metric_card(col, title: str, m: dict, capital: float) -> None:
    with col:
        st.markdown(f"**{title}**")
        st.metric("Directional Accuracy", f"{m['directional_acc']*100:.1f}%")
        st.metric("Cumulative Return", f"{m['cum_profit']*100:+.2f}%", f"{m['cum_profit']*capital:+,.0f} PHP")
        st.metric("Geometric Return", f"{m['cum_profit_geo']*100:+.2f}%", f"{m['cum_profit_geo']*capital:+,.0f} PHP")
        st.metric("Sharpe-like Ratio", f"{m['sharpe_like']:.3f}")
        st.metric("RMSE", f"{m['rmse']:.5f}")
        if m.get("degenerate"):
            st.caption("⚠️ never changes side over the test window")


def comparison_figure(mse: dict, pa: dict):
    specs = [
        ("directional_acc", "Directional Accuracy", "{:.1%}"),
        ("cum_profit", "Cumulative Return", "{:.1%}"),
        ("cum_profit_geo", "Geometric Return", "{:.1%}"),
        ("sharpe_like", "Sharpe-like Ratio", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), facecolor="#fcfcfb")
    for ax, (key, label, fmt) in zip(axes, specs):
        ax.set_facecolor("#fcfcfb")
        vals = [mse[key], pa[key]]
        bars = ax.bar(["MSE", "Profit-Aware"], vals, width=0.55, color=[COLOR_MSE, COLOR_PA])
        ax.axhline(0, color=INK, linewidth=0.8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=10, color=INK)
        for b, v in zip(bars, vals):
            ax.annotate(
                fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK,
            )
    fig.tight_layout()
    return fig


def render_trade_log(container, m: dict) -> None:
    with container:
        summary = m["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wins / Losses", f"{summary['wins']} / {summary['losses']}")
        c2.metric("Best trade", f"{summary['best_trade']*100:+.2f}%", summary["best_date"] or "-")
        c3.metric("Worst trade", f"{summary['worst_trade']*100:+.2f}%", summary["worst_date"] or "-")
        c4.metric("Final geometric balance", f"{summary['final_geo_balance']:,.0f} PHP")

        df = pd.DataFrame(m["records"])
        df["side"] = np.where(df["signal"] > 0, "LONG", np.where(df["signal"] < 0, "SHORT", "FLAT"))
        df["return"] *= 100
        df.index = range(1, len(df) + 1)
        df = df[["date", "return", "pred", "signal", "side", "trade_pnl", "add_balance", "geo_balance"]]
        st.dataframe(
            df,
            use_container_width=True,
            height=420,
            column_config={
                "date": st.column_config.TextColumn("Date"),
                "return": st.column_config.NumberColumn("Return", format="%.2f%%"),
                "pred": st.column_config.NumberColumn("Predicted Return", format="%.4f"),
                "signal": st.column_config.NumberColumn("Signal", format="%.3f"),
                "side": st.column_config.TextColumn("Side"),
                "trade_pnl": st.column_config.NumberColumn("Trade P&L (PHP)", format="%.0f"),
                "add_balance": st.column_config.NumberColumn("Additive Balance (PHP)", format="%.0f"),
                "geo_balance": st.column_config.NumberColumn("Geometric Balance (PHP)", format="%.0f"),
            },
        )


st.title("GRU Stock Predictor — MSE vs Profit-Aware")
st.caption(
    "Pick a ticker from the thesis benchmark universe, train both loss variants live "
    "on the identical chronological split, and compare their test-window trade logs."
)

catalog = ticker_catalog()
labels = [f"{c['symbol']}  ·  {c['market']} / {c['tier']}" for c in catalog]

with st.sidebar:
    st.header("Setup")
    idx = st.selectbox("Ticker", options=range(len(catalog)), format_func=lambda i: labels[i])
    symbol = catalog[idx]["symbol"]

    with st.expander("Advanced settings"):
        sequence_length = st.number_input("Sequence length", 5, 120, 30)
        epochs = st.slider("Epochs", 5, 100, 50, step=5)
        hidden_size = st.number_input("Hidden size", 8, 256, 64, step=8)
        num_layers = st.number_input("GRU layers", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.6, 0.2, step=0.05)
        capital = st.number_input("Capital (PHP)", 1_000.0, 10_000_000.0, 100_000.0, step=10_000.0)
        transaction_cost = st.number_input("Transaction cost rate", 0.0, 0.05, 0.001, step=0.0005, format="%.4f")
        loss_lambda = st.number_input("Profit-aware MSE weight (lambda)", 0.0, 5.0, 0.7, step=0.1)
        seed = st.number_input("Random seed", 0, 10_000, 42)

    run_clicked = st.button("Run", type="primary", use_container_width=True)

cfg = dict(
    start="2018-01-01",
    sequence_length=int(sequence_length),
    epochs=int(epochs),
    hidden_size=int(hidden_size),
    num_layers=int(num_layers),
    dropout=float(dropout),
    capital=float(capital),
    transaction_cost=float(transaction_cost),
    loss_lambda=float(loss_lambda),
    output_cap_std=5.0,
    lr=1e-3,
    weight_decay=1e-3,
    seed=int(seed),
    alpha=None,
)

if run_clicked:
    with st.spinner(f"Training MSE and Profit-Aware GRUs on {symbol} ({epochs} epochs each)..."):
        t0 = time.time()
        st.session_state["result"] = run_comparison(symbol, cfg)
        st.session_state["elapsed"] = time.time() - t0

result = st.session_state.get("result")
if result is None:
    st.info("Choose a ticker in the sidebar and click **Run**.")
    st.stop()

symbol = result["symbol"]
split = result["split"]
mse, pa = result["mse"], result["pa"]
capital = result["capital"]

st.subheader(symbol)

info_cols = st.columns(4)
info_cols[0].metric("Test Window Start", format_date(split.dates_test[0]))
info_cols[1].metric("Test Window End", format_date(split.dates_test[-1]))
info_cols[2].metric("Trading Days", f"{len(split.dates_test)}")
info_cols[3].metric("Alpha", f"{result['alpha']:.4f}")

st.caption(f"Trained in {st.session_state.get('elapsed', 0):.1f}s on {_device()}")

col_mse, col_pa, col_bh = st.columns(3)
metric_card(col_mse, "MSE Baseline", mse, capital)
metric_card(col_pa, "Profit-Aware", pa, capital)
with col_bh:
    st.markdown("**Buy & Hold (reference)**")
    st.caption("Always long, no trading costs — context, not a scored baseline.")
    st.metric(
        "Cumulative Return",
        f"{result['buy_hold_return']*100:+.2f}%",
        f"{result['buy_hold_return']*capital:+,.0f} PHP",
    )

st.markdown("### MSE vs Profit-Aware at a glance")
st.pyplot(comparison_figure(mse, pa))

st.markdown("### Balance over the test window (geometric compounding)")
dates = [format_date(d) for d in split.dates_test]
chart_df = pd.DataFrame(
    {
        "MSE": [r["geo_balance"] for r in mse["records"]],
        "Profit-Aware": [r["geo_balance"] for r in pa["records"]],
        "Buy & Hold": result["buy_hold_balance"],
    },
    index=pd.Index(dates, name="Date"),
)
st.line_chart(chart_df, color=[COLOR_MSE, COLOR_PA, COLOR_REF])

st.markdown("### Trade log")
tab_mse, tab_pa = st.tabs(["MSE Baseline", "Profit-Aware"])
render_trade_log(tab_mse, mse)
render_trade_log(tab_pa, pa)
