"""Rebuild the retained Chapter 3 diagrams for the three loss components."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
INK, EDGE = "#192738", "#6687a0"


def box(ax, x, y, w, h, text, color, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", lw=1.5,
                               edgecolor=EDGE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontsize=size)


def build():
    with plt.rc_context({"font.family": "DejaVu Serif", "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, 10.8))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
        ax.text(.5, .97, "PSEi Profit-Aware GRU Experimental Workflow", ha="center", weight="bold", fontsize=14)
        rows = [
            ("START", "#e0f0e9"),
            ("Fixed PSEi cache → six past-only features\n30-day inputs → four expanding-window folds", "#eaf0f7"),
            ("Shared two-layer GRU and tanh positions\n50 epochs; seed 42; MSE calibration λ = 0.7", "#eaf0f7"),
            ("SOP 1: MSE versus profit-aware loss\nNegative mean simulated net return + λ MSE", "#fff0dc"),
            ("SOP 2: profit-aware versus + downside penalty\nγ mean[max(0, −net return)²]\nSelect positive γ on prior validation data only", "#fff0dc"),
            ("SOP 3: cost omitted versus included in loss\nAsymmetric cost on position changes\nBuy 0.296%; sell 0.396%; common evaluation costs", "#fff0dc"),
            ("Evaluate held-out return, downside risk, forecast error,\nturnover and exposure; compare constant controls", "#e4f1e6"),
            ("Save aligned daily outputs, fold tables and figures", "#e4f1e6"),
            ("END", "#e0f0e9")]
        ys = [.87, .77, .66, .55, .43, .30, .18, .08, .005]
        hs = [.05, .07, .07, .07, .09, .09, .07, .05, .045]
        for i, ((text, color), y, h) in enumerate(zip(rows, ys, hs)):
            box(ax, .08, y, .84, h, text, color, 12)
            if i:
                ax.add_patch(FancyArrowPatch((.5, ys[i-1]-.01), (.5, y+h+.012),
                                             arrowstyle="-|>", mutation_scale=13, color="#596774"))
        fig.savefig(ROOT / "docs/figure_3_1.png", dpi=210, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.8, 5.3))
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
        ax.text(.5, .94, "PSEi Input–Process–Output Architecture", ha="center", weight="bold", fontsize=17)
        labels = [
            "INPUT\n\nFixed PSEi cache\nDaily OHLCV\nClose-to-close returns\nEarlier data only",
            "PROCESS\n\nSix past-only features\n30-day input windows\nFour expanding folds\nProfit / downside / cost\nValidation-only γ",
            "OUTPUT\n\nNext-day forecasts\nContinuous positions\nNet return / downside\nForecast / turnover\nSaved tables / charts"]
        for x, label, color in zip((.015, .365, .715), labels, ("#eaf0f7", "#fff0dc", "#e4f1e6")):
            box(ax, x, .13, .27, .63, label, color, 12)
        for x in (.285, .635):
            ax.add_patch(FancyArrowPatch((x+.015, .45), (x+.065, .45), arrowstyle="-|>",
                                         mutation_scale=17, color="#596774"))
        fig.savefig(ROOT / "docs/figure_3_2.png", dpi=210, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    build()
