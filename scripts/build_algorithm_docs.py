"""Build the detailed algorithm's Markdown and editable HTML from one source."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/algorithm-with-sop-2.json"


def load():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if [s["number"] for s in data["steps"]] != list(range(12)):
        raise ValueError("Expected the original step numbers 0 through 11")
    return data


def paragraph(text, *, bold=False, formula=False, label=False):
    style = (
        "font-family:'Times New Roman';font-size:12pt;"
        f"line-height:{'135%' if formula else '200%'};margin:0 0 6pt 0;"
        "text-indent:0;"
        f"text-align:{'center' if formula else 'left'};"
        f"font-weight:{'bold' if bold or label else 'normal'};"
        f"color:{'#b00000' if label else '#000000'};"
    )
    span_style = ' style="background-color:#ffff00;"' if label else ""
    return f'<p style="{style}"><span{span_style}>{html.escape(text)}</span></p>'


def as_html(data):
    out = [paragraph(data["title"], bold=True)]
    for step in data["steps"]:
        title = f'Step {step["number"]}: {step["title"]}'
        if step["number"] in (0, 11):
            out.append(paragraph(title, bold=True))
            continue
        body = [paragraph(title, bold=True), paragraph("Given: " + step["given"])]
        body += [paragraph(text, formula=kind == "equation") for kind, text in step["content"]]
        body.append(paragraph(step["objective"], label=True))
        out.append(
            '<table style="width:100%;border-collapse:collapse;page-break-inside:avoid;">'
            '<tbody><tr><td style="border:2px solid #d00000;padding:8pt;vertical-align:top;">'
            + "".join(body) + '</td></tr></tbody></table>'
            '<p style="font-size:6pt;line-height:100%;margin:0;">&nbsp;</p>'
        )
    return "".join(out)


def as_markdown(data):
    out = ["# Detailed proposed algorithm with SOP 2", "", data["note"], "",
           "This text and the editable HTML replacement are generated from `algorithm-with-sop-2.json`. See `figure-alignment-2026-10-28.md` for the live-document synchronization status.", ""]
    for step in data["steps"]:
        out += [f'## Step {step["number"]} {step["title"]}', ""]
        if step["given"]:
            out += ["**Given:** " + step["given"], ""]
        for kind, text in step["content"]:
            out += [f"`{text}`" if kind == "equation" else text, ""]
        if step["objective"]:
            out += ["**" + step["objective"] + "**", ""]
    out += ["## Implementation sources", "",
            "- `utils/preprocessing.py`: cleaning, six features, sequences, chronological folds.",
            "- `models/gru_model.py`: two-layer GRU, layer normalization, bounded output head.",
            "- `utils/trading.py`: training-derived alpha, smooth position, asymmetric position-change cost.",
            "- `training/train.py`: MSE versus combined loss, carried batch position, AdamW, final-epoch training.",
            "- [PyTorch 2.12 GRU](https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html)",
            "- [PyTorch 2.12 AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    data = load()
    destination = ROOT / "docs/algorithm-with-sop-2.md"
    destination.write_text(as_markdown(data), encoding="utf-8")
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(as_html(data), encoding="utf-8")
    print(f"Built 12 steps from {SOURCE.name}")


if __name__ == "__main__":
    main()
