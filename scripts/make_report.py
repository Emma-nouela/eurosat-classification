"""Build the EuroSAT analysis report PDF.

Assembles a single PDF deliverable (``EuroSAT_Analysis_Report.pdf``) that presents the
methodology, results (tables + diagrams) and model comparison required by the assignment.
It embeds the figures already generated under ``results/`` and renders summary/per-class
tables from the committed metrics JSON files.

Dependencies are limited to matplotlib + Pillow (already in requirements.txt) so the report
can be regenerated anywhere with::

    python scripts/make_report.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "EuroSAT_Analysis_Report.pdf"

# A4 portrait in inches.
A4 = (8.27, 11.69)
MARGIN = 0.07  # fraction of figure used as side margin

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def load_json(name: str) -> dict:
    with open(RESULTS / name, encoding="utf-8") as fh:
        return json.load(fh)


def new_page(pdf: PdfPages):
    fig = plt.figure(figsize=A4)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def add_heading(fig, text, y=0.95, size=17):
    fig.text(MARGIN, y, text, fontsize=size, fontweight="bold", va="top",
             color="#0b3d63")
    fig.text(MARGIN, y - 0.018, "", fontsize=1)  # spacer
    # underline
    fig.add_artist(plt.Line2D([MARGIN, 1 - MARGIN], [y - 0.022, y - 0.022],
                              color="#0b3d63", linewidth=1.2, transform=fig.transFigure))


def add_paragraph(fig, text, y, size=10.5, x=MARGIN, color="#1a1a1a",
                  weight="normal", linespacing=1.45):
    fig.text(x, y, text, fontsize=size, va="top", ha="left", wrap=True,
             color=color, fontweight=weight, linespacing=linespacing)


def add_table(fig, col_labels, rows, y_top, y_bottom, col_widths=None,
              header_color="#0b3d63", fontsize=9.0, title=None):
    ax = fig.add_axes([MARGIN, y_bottom, 1 - 2 * MARGIN, y_top - y_bottom])
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left",
                     color="#0b3d63", pad=10)
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3f7")
    return ax


def add_image(fig, rect, path, title=None):
    """rect = [left, bottom, width, height] in figure fraction; image is
    centered and scaled to preserve aspect ratio inside rect."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0b3d63")
    img = mpimg.imread(str(path))
    ax.imshow(img)
    return ax


def footer(fig, page_no):
    fig.text(0.5, 0.025, f"EuroSAT Land-Use Classification — Analysis Report   |   page {page_no}",
             fontsize=8, ha="center", color="#888888")


# --------------------------------------------------------------------------------------
# Page builders
# --------------------------------------------------------------------------------------

def page_title(pdf, page_no):
    fig = new_page(pdf)
    fig.text(0.5, 0.74, "EuroSAT Land-Use Classification",
             fontsize=26, fontweight="bold", ha="center", color="#0b3d63")
    fig.text(0.5, 0.69, "Custom CNN (from scratch)  vs.  ResNet50 (transfer learning)",
             fontsize=14, ha="center", color="#333333")
    fig.add_artist(plt.Line2D([0.28, 0.72], [0.655, 0.655], color="#0b3d63",
                              linewidth=1.5, transform=fig.transFigure))
    subtitle = (
        "MSc in Artificial Intelligence & Visual Computing (AIVC)\n"
        "University of West Attica\n"
        "Assignment (Ergasia) 2026 — Image Classification"
    )
    fig.text(0.5, 0.60, subtitle, fontsize=12, ha="center", va="top",
             color="#222222", linespacing=1.6)

    fig.text(0.5, 0.40,
             "A comparative study of two deep-learning approaches for land-use / land-cover\n"
             "classification on 27,000 Sentinel-2 satellite tiles across 10 classes.",
             fontsize=11, ha="center", va="top", color="#444444", linespacing=1.5,
             style="italic")

    fig.text(0.5, 0.16, f"Author: vigoroth\n{date.today().isoformat()}",
             fontsize=11, ha="center", va="top", color="#222222", linespacing=1.6)
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def page_idea(pdf, page_no):
    fig = new_page(pdf)
    add_heading(fig, "1. Problem description")
    add_paragraph(
        fig,
        "EuroSAT is a benchmark dataset of Sentinel-2 satellite images for land-use / land-cover\n"
        "(LULC) classification. It contains 27,000 RGB images of 64×64 pixels, evenly spread\n"
        "across 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,\n"
        "PermanentCrop, Residential, River and SeaLake. The goal of this project is to train, analyse\n"
        "and compare at least two different deep-learning models for this classification task,\n"
        "evaluating their performance, capabilities and limitations.",
        y=0.90,
    )

    add_heading(fig, "2. The Idea (methodology)", y=0.745)
    add_paragraph(
        fig,
        "To contrast two of the approaches suggested in the assignment, we implement and compare:",
        y=0.695,
    )
    rows = [
        ["Custom CNN", "Trained from scratch",
         "Compact VGG-style net (4 conv blocks)\nbuilt for 64×64 tiles — a lightweight baseline."],
        ["ResNet50", "Transfer learning\n+ fine-tuning",
         "ImageNet-pretrained backbone with a new\nclassifier head, fully fine-tuned."],
    ]
    add_table(fig, ["Model", "Approach", "Rationale"], rows,
              y_top=0.66, y_bottom=0.50, col_widths=[0.18, 0.22, 0.60])

    add_paragraph(
        fig,
        "Pipeline (identical for both models, for a fair comparison):",
        y=0.46, weight="bold",
    )
    add_paragraph(
        fig,
        "1.  Data — download EuroSAT (Zenodo) and build a reproducible stratified 70/15/15\n"
        "     train/validation/test split (seed = 42) → 18,900 / 4,050 / 4,050 images.\n"
        "2.  Augmentation — random horizontal/vertical flips and ±15° rotation (satellite tiles are\n"
        "     orientation-agnostic). The CNN uses native 64×64 with EuroSAT normalization; ResNet50\n"
        "     resizes to 224×224 with ImageNet normalization.\n"
        "3.  Training — CrossEntropyLoss, Adam, ReduceLROnPlateau scheduler; best checkpoint chosen\n"
        "     by validation accuracy.\n"
        "4.  Hyperparameter optimization — Optuna (5 trials) tunes learning rate, dropout, batch size\n"
        "     and optimizer.\n"
        "5.  Evaluation — accuracy + per-class precision/recall/F1, confusion matrix and a\n"
        "     sample-prediction grid on the held-out test set.",
        y=0.425, size=10,
    )
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def page_summary_tables(pdf, page_no, cnn, resnet, opt_cnn, opt_resnet):
    fig = new_page(pdf)
    add_heading(fig, "3. Analysis — overall results")

    cnn_acc = cnn["accuracy"]
    res_acc = resnet["accuracy"]
    cnn_f1 = cnn["report"]["macro avg"]["f1-score"]
    res_f1 = resnet["report"]["macro avg"]["f1-score"]

    rows = [
        ["Custom CNN", "from scratch", f"{cnn_acc:.4f}", f"{cnn_f1:.4f}",
         "457,738 (≈0.46M)", "~5.3 s"],
        ["ResNet50", "transfer learning\n+ fine-tuning", f"{res_acc:.4f}", f"{res_f1:.4f}",
         "23,528,522 (≈23.5M)", "~125 s"],
    ]
    add_table(
        fig,
        ["Model", "Approach", "Test acc.", "Macro F1", "Trainable params", "Epoch time"],
        rows, y_top=0.90, y_bottom=0.74,
        col_widths=[0.16, 0.18, 0.13, 0.12, 0.24, 0.13],
        title="Summary — test-set performance",
    )

    add_paragraph(
        fig,
        f"ResNet50 reaches the highest test accuracy ({res_acc*100:.1f}%), about "
        f"+{(res_acc-cnn_acc)*100:.1f} points over the\nfrom-scratch CNN ({cnn_acc*100:.1f}%), "
        "and converges to >0.98 validation accuracy in a single fine-tuning\nepoch — confirming the "
        "value of ImageNet transfer learning on satellite imagery. The Custom\nCNN remains a strong, "
        "~51× lighter baseline.",
        y=0.70, size=10.5,
    )

    o_rows = [
        ["Custom CNN", f"{opt_cnn['lr']:.2e}", f"{opt_cnn['dropout']:.3f}",
         str(opt_cnn["batch_size"]), opt_cnn["optimizer"], f"{opt_cnn['val_acc']:.3f}"],
        ["ResNet50", f"{opt_resnet['lr']:.2e}", f"{opt_resnet['dropout']:.3f}",
         str(opt_resnet["batch_size"]), opt_resnet["optimizer"], f"{opt_resnet['val_acc']:.3f}"],
    ]
    add_table(
        fig,
        ["Model", "lr", "dropout", "batch", "optimizer", "val acc (5-epoch trial)"],
        o_rows, y_top=0.55, y_bottom=0.40,
        col_widths=[0.18, 0.14, 0.13, 0.10, 0.16, 0.29],
        title="Optuna best configuration (5 trials each)",
    )
    add_paragraph(
        fig,
        "The search optimizes validation accuracy over short 5-epoch trials. Full per-trial tables are\n"
        "committed under results/optuna_cnn_trials.csv and results/optuna_resnet_trials.csv.",
        y=0.36, size=9.5, color="#555555",
    )
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def page_image_pair(pdf, page_no, heading, caption, img_cnn, img_resnet, layout="row"):
    fig = new_page(pdf)
    add_heading(fig, heading)
    if layout == "row":
        # two stacked wide images (e.g. curves are 3:1)
        add_image(fig, [MARGIN, 0.66, 1 - 2 * MARGIN, 0.20], img_cnn, title="Custom CNN")
        add_image(fig, [MARGIN, 0.38, 1 - 2 * MARGIN, 0.20], img_resnet, title="ResNet50")
    elif layout == "side":
        add_image(fig, [MARGIN, 0.46, 0.42, 0.40], img_cnn, title="Custom CNN")
        add_image(fig, [0.51, 0.46, 0.42, 0.40], img_resnet, title="ResNet50")
    elif layout == "tall":
        add_image(fig, [MARGIN, 0.30, 0.42, 0.58], img_cnn, title="Custom CNN")
        add_image(fig, [0.51, 0.30, 0.42, 0.58], img_resnet, title="ResNet50")
    if caption:
        add_paragraph(fig, caption, y=0.30 if layout == "side" else 0.26,
                      size=10, color="#444444")
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def page_perclass(pdf, page_no, cnn, resnet):
    fig = new_page(pdf)
    add_heading(fig, "Per-class metrics (test set)")
    rows = []
    for cls in CLASS_NAMES:
        c = cnn["report"][cls]
        r = resnet["report"][cls]
        rows.append([
            cls,
            f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1-score']:.3f}",
            f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}",
        ])
    rows.append([
        "macro avg",
        f"{cnn['report']['macro avg']['precision']:.3f}",
        f"{cnn['report']['macro avg']['recall']:.3f}",
        f"{cnn['report']['macro avg']['f1-score']:.3f}",
        f"{resnet['report']['macro avg']['precision']:.3f}",
        f"{resnet['report']['macro avg']['recall']:.3f}",
        f"{resnet['report']['macro avg']['f1-score']:.3f}",
    ])
    ax = add_table(
        fig,
        ["Class", "CNN P", "CNN R", "CNN F1", "ResNet P", "ResNet R", "ResNet F1"],
        rows, y_top=0.90, y_bottom=0.40,
        col_widths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        fontsize=9.0,
    )
    # bold the macro-avg row
    tbl = [a for a in ax.get_children() if hasattr(a, "get_celld")]
    add_paragraph(
        fig,
        "Both models are strongest on visually distinct classes (Forest, SeaLake, Residential) and\n"
        "weakest on the spectrally similar crop/vegetation classes. For the Custom CNN, PermanentCrop\n"
        "(recall 0.89), River (0.93) and HerbaceousVegetation (0.95) are the hardest; ResNet50 is much\n"
        "flatter, keeping every class ≥ 0.97 recall.",
        y=0.36, size=10, color="#333333",
    )
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def page_comparison(pdf, page_no):
    fig = new_page(pdf)
    add_heading(fig, "4. Comparison, capabilities & limitations")

    add_paragraph(fig, "Custom CNN (from scratch)", y=0.90, size=12, weight="bold",
                  color="#0b3d63")
    add_paragraph(
        fig,
        "+  Lightweight (457K params, ~51× fewer than ResNet50), ~5 s/epoch, low memory — runs on\n"
        "    modest hardware.\n"
        "+  Operates on native 64×64 tiles (no upscaling needed).\n"
        "+  Still reaches a strong 96.3% test accuracy from scratch.\n"
        "−  Lower accuracy ceiling; must learn all visual features from limited data.\n"
        "−  More sensitive to augmentation / regularization choices.",
        y=0.865, size=10.5,
    )

    add_paragraph(fig, "ResNet50 (transfer learning + fine-tuning)", y=0.70, size=12,
                  weight="bold", color="#0b3d63")
    add_paragraph(
        fig,
        "+  Highest accuracy (98.7% test, +2.4 pts over the CNN); ImageNet features transfer well to\n"
        "    satellite imagery.\n"
        "+  Converges fast — >0.98 validation accuracy after a single fine-tuning epoch.\n"
        "−  ~23.5M parameters; ~24× slower per epoch (~125 s), heavier memory and inference cost.\n"
        "−  Requires upscaling 64×64 → 224×224 and ImageNet normalization.",
        y=0.665, size=10.5,
    )

    add_paragraph(fig, "Shared failure mode", y=0.52, size=12, weight="bold",
                  color="#0b3d63")
    add_paragraph(
        fig,
        "Both models confuse the same semantically/visually similar land-cover classes —\n"
        "PermanentCrop ↔ AnnualCrop and HerbaceousVegetation ↔ Pasture — which is expected for\n"
        "low-resolution 64×64 satellite tiles where texture cues are limited (see confusion matrices).",
        y=0.485, size=10.5,
    )

    add_paragraph(fig, "Conclusion", y=0.37, size=12, weight="bold", color="#0b3d63")
    add_paragraph(
        fig,
        "If accuracy is the priority and a GPU is available, ResNet50 with transfer learning is the\n"
        "clear choice. Where compute, memory or latency are constrained, the Custom CNN delivers\n"
        "96%+ accuracy at a fraction of the cost — a favourable accuracy/efficiency trade-off.\n"
        "All randomness is seeded (SEED = 42), so the split and training are reproducible.",
        y=0.335, size=10.5,
    )
    footer(fig, page_no)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    cnn = load_json("cnn_metrics.json")
    resnet = load_json("resnet_metrics.json")
    opt_cnn = load_json("optuna_cnn_best_params.json")
    opt_resnet = load_json("optuna_resnet_best_params.json")

    plt.rcParams["font.family"] = "DejaVu Sans"

    with PdfPages(OUTPUT) as pdf:
        n = 1
        page_title(pdf, n); n += 1
        page_idea(pdf, n); n += 1
        page_summary_tables(pdf, n, cnn, resnet, opt_cnn, opt_resnet); n += 1
        page_image_pair(pdf, n, "Training curves (Loss / Accuracy)",
                        "Per-epoch training and validation loss and accuracy. ResNet50 starts near "
                        "its ceiling thanks to\npretrained features; the CNN improves steadily over "
                        "the full schedule.",
                        RESULTS / "cnn_curves.png", RESULTS / "resnet_curves.png", layout="row")
        n += 1
        page_image_pair(pdf, n, "Confusion matrices (test set)",
                        "Counts of true vs. predicted classes. ResNet50's off-diagonal mass is much "
                        "lighter, concentrated\non the crop/vegetation classes.",
                        RESULTS / "cnn_confusion_matrix.png",
                        RESULTS / "resnet_confusion_matrix.png", layout="side")
        n += 1
        page_perclass(pdf, n, cnn, resnet); n += 1
        page_image_pair(pdf, n, "Sample predictions (green = correct, red = wrong)",
                        "Random held-out test tiles with predicted labels.",
                        RESULTS / "cnn_predictions.png",
                        RESULTS / "resnet_predictions.png", layout="tall")
        n += 1
        page_image_pair(pdf, n, "Optuna hyperparameter search history",
                        "Validation accuracy per trial and the running best over the 5-trial search.",
                        RESULTS / "optuna_cnn_history.png",
                        RESULTS / "optuna_resnet_history.png", layout="side")
        n += 1
        page_comparison(pdf, n); n += 1

    print(f"Wrote {OUTPUT}  ({OUTPUT.stat().st_size/1024:.0f} KB, {n-1} pages)")


if __name__ == "__main__":
    main()
