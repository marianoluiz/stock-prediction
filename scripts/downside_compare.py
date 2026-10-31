"""Matched CPU downside-loss ablation on the four existing PSEi folds.

Train: .venv/Scripts/python.exe scripts/downside_compare.py --train
Replot saved numerical results: same command without --train.

Protocol: docs/downside-protocol.md. No held-out return selects gamma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from models.gru_model import GRUReturnPredictor
from scripts.significance import newey_west_test
from training.train import TrainingConfig, fit, run_epoch
from utils.pipeline import prepare_walk_forward, cache_path_for
from utils.trading import PH_BUY_COST, PH_SELL_COST, calibrate_alpha, position_change_cost
from utils.metrics import sharpe_like

KAPPAS = (0.25, 1.0, 4.0)
LOSS_LAMBDA = 0.7
NAMES = {"profit-aware": "Profit-aware", "profit-aware-downside": "Profit-aware + downside"}


def simulate(signal, actual):
    signal, actual = np.asarray(signal, dtype=float), np.asarray(actual, dtype=float)
    prev = np.r_[0.0, signal[:-1]]
    cost = position_change_cost(signal, prev, PH_BUY_COST, PH_SELL_COST)
    return signal * actual - cost, cost, np.abs(signal - prev)


def validation_utility(pred, actual, signal, alpha):
    """Common scoring rule; gamma used to train the candidate is NOT its score.

    Reference kappa=1 for every candidate. Only validation inputs are accepted.
    """
    net, _, _ = simulate(signal, actual)
    return float(net.mean() - LOSS_LAMBDA * np.mean((pred - actual) ** 2)
                 - alpha * np.mean(np.maximum(-net, 0.0) ** 2))


def select_candidate(rows):
    """Maximum common validation utility; smallest kappa breaks exact ties."""
    return min(rows, key=lambda row: (-row["validation_utility"], row["kappa"]))


def metrics(pred, actual, signal):
    pred, actual, signal = (np.asarray(values, dtype=float) for values in (pred, actual, signal))
    net, cost, turnover = simulate(signal, actual)
    lpm2 = float(np.mean(np.maximum(-net, 0.0) ** 2))
    nonzero = np.sign(signal[np.abs(signal) > 1e-8])
    switches = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    balance = np.r_[1.0, np.cumprod(1.0 + net)]
    return {
        "additive_return": float(net.sum()),
        "geometric_return": float(np.prod(1.0 + net) - 1.0),
        "sharpe_like": sharpe_like(net),
        "downside_lpm2": lpm2,
        "downside_deviation_daily": lpm2 ** 0.5,
        "negative_return_sum": float(np.maximum(-net, 0.0).sum()),
        "loss_day_share": float((net < 0).mean()),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "directional_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
        "turnover": float(turnover.mean()),
        "cost_sum": float(cost.sum()),
        "mean_position": float(np.mean(signal)),
        "mean_abs_position": float(np.mean(np.abs(signal))),
        "near_flat_share": float(np.mean(np.abs(signal) < 0.01)),
        "short_share": float(np.mean(signal < 0)),
        "sign_switches": switches,
        "constant_side": int(len(nonzero) > 0 and len(np.unique(nonzero)) == 1),
        "max_drawdown": float(np.min(balance / np.maximum.accumulate(balance) - 1)),
        "worst_net_day": float(net.min()),
    }


def train_candidate(data, alpha, kappa, epochs, device):
    torch.manual_seed(42)
    model = GRUReturnPredictor(input_size=data.split.x_train.shape[-1], hidden_size=64,
                              num_layers=2, dropout=0.2, output_scale=5.0 / alpha).to(device)
    loss_type = "profit-aware-downside" if kappa else "profit-aware"
    config = TrainingConfig(loss_type=loss_type, alpha=alpha, loss_lambda=LOSS_LAMBDA,
                            downside_gamma=kappa * alpha, buy_cost_rate=PH_BUY_COST,
                            sell_cost_rate=PH_SELL_COST, learning_rate=0.001,
                            weight_decay=0.001, batch_size=64, epochs=epochs)
    history = fit(model, data.train_loader, data.val_loader, config, device)
    val = run_epoch(model, data.val_loader, None, alpha, PH_BUY_COST, PH_SELL_COST,
                    device, loss_type=loss_type, loss_lambda=LOSS_LAMBDA,
                    downside_gamma=kappa * alpha)
    return model, history, val


def train(out, epochs):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    # Exact fixed cache, identical to the original paper's four windows.
    cache = ROOT / cache_path_for("PSE:PSEI", "2018-01-01", None)
    if not cache.exists():
        raise FileNotFoundError("The committed PSEi cache is required; no live data substitution")
    expected = "35bb13112fc537d62697042bb7dd27803816f51254cadad2822ed0e5913d4972"
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("PSEi cache differs from the paper protocol; inspect before retraining")
    manifest = {"protocol": "downside-v1", "cache_sha256": digest, "kappas": KAPPAS,
                "lambda": LOSS_LAMBDA, "gamma_rule": "kappa / std(y_train), ddof=0",
                "selection": "max common validation utility with reference kappa=1; smallest kappa ties",
                "epochs": epochs, "seed": 42, "device": "cpu", "threads": 4,
                "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                "buy_cost": PH_BUY_COST, "sell_cost": PH_SELL_COST, "folds": []}
    out.mkdir(parents=True, exist_ok=True)
    # Record the protocol before training or looking at this experiment's test outcomes.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    folds = prepare_walk_forward("PSE:PSEI", "2018-01-01", None, 30, 64, n_folds=4)
    summaries, days, validations, histories = [], [], [], []
    for data in folds:
        alpha = calibrate_alpha(data.split.y_train)
        print(f"\nFOLD {data.fold}: training gamma=0 control and three positive candidates", flush=True)
        base_model, base_history, base_val = train_candidate(data, alpha, 0.0, epochs, device)
        models, val_rows = {}, []
        histories.extend({"fold": data.fold, "kappa": 0.0, "epoch": i + 1,
                          **{key: values[i] for key, values in base_history.items()}}
                         for i in range(epochs))
        for kappa in KAPPAS:
            model, history, val = train_candidate(data, alpha, kappa, epochs, device)
            score = validation_utility(val["pred_np"], val["actual_np"], val["signal_np"], alpha)
            row = {"fold": data.fold, "kappa": kappa, "gamma": kappa * alpha,
                   "validation_utility": score,
                   "validation_mean_position": float(val["signal_np"].mean()),
                   **{f"validation_{key}": value for key, value in metrics(
                       val["pred_np"], val["actual_np"], val["signal_np"]).items()}}
            val_rows.append(row)
            models[kappa] = model
            histories.extend({"fold": data.fold, "kappa": kappa, "epoch": i + 1,
                              **{key: values[i] for key, values in history.items()}}
                             for i in range(epochs))
        chosen = select_candidate(val_rows)
        kappa = chosen["kappa"]
        for row in val_rows:
            row["selected"] = row["kappa"] == kappa
        validations.extend(val_rows)
        split_info = {"fold": data.fold, "alpha": alpha, "selected_kappa": kappa,
                      "selected_gamma": kappa * alpha,
                      "train_n": len(data.split.y_train), "val_n": len(data.split.y_val),
                      "test_n": len(data.split.y_test),
                      "validation_end": str(data.split.dates_val[-1])[:10],
                      "test_start": str(data.split.dates_test[0])[:10],
                      "test_end": str(data.split.dates_test[-1])[:10]}
        assert split_info["validation_end"] < split_info["test_start"]
        manifest["folds"].append(split_info)
        # Selection is written BEFORE any held-out evaluation.
        pd.DataFrame(validations).to_csv(out / "validation.csv", index=False)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name, model, weight in (("profit-aware", base_model, 0.0),
                                    ("profit-aware-downside", models[kappa], kappa)):
            test = run_epoch(model, data.test_loader, None, alpha, PH_BUY_COST, PH_SELL_COST,
                             device, loss_type=name, loss_lambda=LOSS_LAMBDA,
                             downside_gamma=weight * alpha)
            pred, actual, signal = test["pred_np"], test["actual_np"], test["signal_np"]
            summaries.append({**split_info, "model": name, "kappa": weight,
                              "gamma": weight * alpha, **metrics(pred, actual, signal)})
            net, cost, turnover = simulate(signal, actual)
            for i, date in enumerate(data.split.dates_test):
                days.append({"fold": data.fold, "date": str(date)[:10], "model": name,
                             "predicted_return": float(pred[i]), "actual_return": float(actual[i]),
                             "position": float(signal[i]), "cost": cost[i],
                             "turnover": turnover[i], "net_return": net[i]})
            # Constant-position diagnostic fixed by VALIDATION exposure, not test exposure.
            exposure = (float(base_val["signal_np"].mean()) if weight == 0 else
                        chosen["validation_mean_position"])
            static_signal = np.full_like(signal, exposure)
            static_pred = np.full_like(pred, np.arctanh(np.clip(exposure, -0.999999, 0.999999)) / alpha)
            summaries.append({**split_info, "model": name + "-validation-static", "kappa": weight,
                              "gamma": weight * alpha, **metrics(static_pred, actual, static_signal)})
        zeros = np.zeros_like(data.split.y_test)
        summaries.append({**split_info, "model": "cash", "kappa": 0.0, "gamma": 0.0,
                          **metrics(zeros, data.split.y_test, zeros)})
        pd.DataFrame(summaries).to_csv(out / "folds.csv", index=False)
        pd.DataFrame(days).to_csv(out / "daily.csv", index=False)
        pd.DataFrame(histories).to_csv(out / "history.csv", index=False)
        print(f"FOLD {data.fold} complete: validation selected kappa={kappa}", flush=True)


def report(out):
    folds = pd.read_csv(out / "folds.csv")
    daily = pd.read_csv(out / "daily.csv")
    assert set(folds.fold) == {0, 1, 2, 3}
    numeric = list(metrics(np.zeros(2), np.zeros(2), np.zeros(2)))
    summary = folds.groupby("model", sort=False)[numeric].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    wide = daily.pivot(index=["fold", "date"], columns="model", values="net_return")
    assert len(wide) == 631 and not wide.isna().any().any()
    a, b = wide["profit-aware-downside"].to_numpy(), wide["profit-aware"].to_numpy()
    rows = []
    for name, diff in (("daily_net_return", a - b),
                       ("daily_downside_lpm2", np.maximum(-a, 0) ** 2 - np.maximum(-b, 0) ** 2)):
        avg, se, statistic, p = newey_west_test(diff)
        half = stats.t.ppf(0.975, len(diff) - 1) * se
        rows.append({"outcome": name, "n": len(diff), "mean_difference": avg,
                     "ci95_lo": avg - half, "ci95_hi": avg + half,
                     "hac_t": statistic, "p_two_sided": p})
    pd.DataFrame(rows).to_csv(out / "significance.csv", index=False)
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))
    for ax, (metric, title, scale, suffix) in zip(axes, (
            ("downside_deviation_daily", "Daily downside deviation", 100, "%"),
            ("geometric_return", "Net geometric return", 100, "%"),
            ("mean_abs_position", "Mean absolute position", 1, ""))):
        for name, offset, color in (("profit-aware", -0.18, "#697b8a"),
                                    ("profit-aware-downside", 0.18, "#17577b")):
            part = folds[folds.model == name].sort_values("fold")
            ax.bar(part.fold + offset + 1, part[metric] * scale, 0.34, color=color, label=NAMES[name])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([1, 2, 3, 4], ["Fold 1", "Fold 2", "Fold 3", "Fold 4"])
        ax.axhline(0, color="#555555", lw=0.6)
        if suffix:
            ax.set_ylabel(suffix)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "downside_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(previous_style)
    lines = ["Four-fold matched CPU downside-loss ablation; lambda=0.7, seed=42.",
             "Positive kappa selected from validation only; gamma=kappa/std(training returns).",
             summary.to_string(index=False), "", pd.DataFrame(rows).to_string(index=False),
             "", "Fold means are descriptive; daily HAC tests use 631 paired days, not 631 independent folds.",
             "Static controls use validation mean position. Cash returns are exactly zero before financing costs."]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results/downside")
    args = parser.parse_args()
    if args.train:
        train(args.out.resolve(), epochs=50)
    report(args.out.resolve())


if __name__ == "__main__":
    main()
