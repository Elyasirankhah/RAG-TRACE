import json
import matplotlib
\
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os
\
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
\
NAVY = "#3C5488"
RED = "#E64B35"
TEAL = "#00A087"
INK = "#222222"
GRID = "#EEF0F3"
\
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
\
\
def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def panel_letter(ax, letter):
    ax.text(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )
def grouped(ax, labels, series, ylabel, ymax, colors):
    """series is a list of (name, values)."""
    x = np.arange(len(labels))
    n = len(series)
    width = 0.78 / n
    for i, (name, values) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            color=colors[i],
            label=name,
            zorder=3,
            linewidth=0,
        )
        for bar, value in zip(bars, values):
            if value == 0:
                text = "0"
            elif value < 1:
                text = f"{value:.2f}".rstrip("0").rstrip(".")
            else:
                text = f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.018,
                text,
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(axis="x", length=0)
    return ax
def ci_dot(ax, y, mean, low, high, color, label=None):
    ax.plot([low, high], [y, y], color=color, lw=1.2, zorder=2)
    ax.scatter([mean], [y], s=22, color=color, zorder=3, linewidths=0, label=label)
def workflow():
    fig, ax = plt.subplots(figsize=(7.35, 1.65))
    ax.set_axis_off()
    \
    def box(x, y, w, h, text):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#C9CED6",
            facecolor="#FFFFFF",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK)
        return patch
    def arrow(x0, y0, x1, y1):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#8A93A3",
            )
        )
    y = 0.35
    h = 0.38
    w = 0.145
    gap = 0.03
    x = 0.02
    \
    b1 = box(x, y, w, h, "ECG\n(PTB-XL)")
    x += w + gap
    b2 = box(x, y, w, h, "Diagnostic model\n(probabilities)")
    x += w + gap
    b3 = box(x, y, w, h, "Retrieval\n(top-3 cards)")
    x += w + gap
    b4 = box(x, y, w, h, "LLM note\n(JSON claims)")
    x += w + gap
    b5 = box(x, y, w, h, "Verification edit\n(withdraw criteria)")
    x += w + gap
    b6 = box(x, y, w, h, "Outcomes\n(3 scores)")
    \
    boxes = [b1, b2, b3, b4, b5, b6]
    for left, right in zip(boxes[:-1], boxes[1:]):
        x0 = left.get_x() + left.get_width()
        y0 = left.get_y() + left.get_height() / 2
        x1 = right.get_x()
        y1 = right.get_y() + right.get_height() / 2
        arrow(x0, y0, x1, y1)
    ax.text(
        0.5,
        0.95,
        "ECG → probabilities → retrieval → note → edit → evaluation",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=INK,
    )
    fig.tight_layout()
    save(fig, "fig_workflow")
def endpoint():
    labels = ["Expert diagnosis\nabsent", "Uni-G rejects\na criterion", "Release score"]
    before = [
        ("gpt-4.1-mini", [10.1, 5.3, 46.8]),
        ("Qwen3-8B", [5.1, 20.8, 65.4]),
    ]
    after = [
        ("gpt-4.1-mini", [10.1, 4.6, 0.0]),
        ("Qwen3-8B", [5.1, 4.7, 0.0]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    grouped(axes[0], labels, before, "Records (%)", 82, [NAVY, RED])
    grouped(axes[1], labels, after, "", 82, [NAVY, RED])
    axes[0].set_title("Keyword note", fontsize=8, pad=8, color=INK)
    axes[1].set_title("After the 12SL withdrawal", fontsize=8, pad=8, color=INK)
    panel_letter(axes[0], "A")
    panel_letter(axes[1], "B")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_endpoint")
def statement():
    labels = ["No retrieval", "Keyword", "12SL withdrawal", "Both algorithms"]
    series = [
        ("gpt-4.1-mini", [7.5, 24.9, 24.6, 22.2]),
        ("Qwen3-8B", [1.6, 34.1, 22.8, 19.6]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    grouped(ax, labels, series, "Records with a statement mismatch (%)", 46, [NAVY, RED])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_statement")
def matched():
    classes = ["Normal", "Infarction", "ST/T change", "Conduction\ndisturbance", "Hypertrophy"]
    recall = [
        ("No cards", [87.5, 67.1, 74.9, 74.4, 61.7]),
        ("14-card constraint", [86.6, 66.3, 74.1, 72.4, 51.4]),
    ]
    summary = ["Precision", "Recall", "F1"]
    micro = [
        ("No cards", [84.1, 77.0, 80.4]),
        ("14-card constraint", [84.4, 75.2, 79.5]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.45, 1]})
    grouped(axes[0], classes, recall, "Class recall (%)", 108, [NAVY, TEAL])
    grouped(axes[1], summary, micro, "Micro score (%)", 108, [NAVY, TEAL])
    axes[0].set_title("Stored superclass", fontsize=8, pad=8, color=INK)
    axes[1].set_title("All labels pooled", fontsize=8, pad=8, color=INK)
    panel_letter(axes[0], "A")
    panel_letter(axes[1], "B")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_matched")
def _pct(value):
    return round(100.0 * float(value), 1)
def _arms(name):
    path = os.path.join(PROJECT, "results", "ptbxl_baseline", name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["arms"]
def _triplet(arms, arm):
    row = arms[arm]
    return [
        _pct(row["diagnosis_absent_from_expert_label"]),
        _pct(row["criterion_rejected_by_unig"]),
        _pct(row["criterion_rejected_by_12sl_or_unchecked"]),
    ]
def _model_bars(ax, labels, series, ylabel, ymax):
    """series is (name, values, color). Models sit on the x-axis."""
    count = len(series)
    x = np.arange(len(labels))
    width = 0.82 / count
    for index, (name, values, color) in enumerate(series):
        offset = (index - (count - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            color=color,
            label=name,
            zorder=3,
            linewidth=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    plt.setp(ax.get_xticklabels(), rotation=32, ha="right")
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(axis="x", length=0)
    return ax
BLUE = "#1A5276"
BLUE2 = "#5DADE2"
GREEN = "#0E6655"
GREEN2 = "#1E8449"
GREEN3 = "#7DCEA0"
MODELS = [
    ("gpt-4.1-mini", "independent_factuality.json", "gpt_heldout"),
    ("Qwen3-8B", "qwen_independent.json", "qwen3_8b"),
    ("Qwen3-14B", "qwen14_independent.json", "qwen3_14b"),
    ("Gemma 3", "gemma3_independent.json", "gemma3_12b"),
    ("MedGemma", "medgemma_independent.json", "medgemma_4b"),
    ("GLM", "glm/heldout_independent.json", "glm_heldout"),
    ("DeepSeek", "deepseek_full_independent.json", "deepseek_r1_8b"),
]
\
\
def _cohort():
    labels = [item[0] for item in MODELS]
    loaded = [_arms(item[1]) for item in MODELS]
    prf_path = os.path.join(PROJECT, "results", "ptbxl_baseline", "diagnosis_prf.json")
    with open(prf_path, encoding="utf-8") as handle:
        prf = json.load(handle)["arms"]
    return labels, loaded, prf
def _style_quad(fig, axes, titles):
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=2)
        ncol = 3 if len(ax.get_legend_handles_labels()[0]) > 2 else 2
        ax.legend(
            frameon=False,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=ncol,
        )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.06, hspace=0.72, wspace=0.20)
def bars_scores():
    """Wide 2x2: three scores before and after withdrawal, plus statement mismatch."""
    labels, loaded, _prf = _cohort()
    keyword = [_triplet(arms, "keyword") for arms in loaded]
    after = [_triplet(arms, "check_12sl") for arms in loaded]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    panels = [
        (0, 20),
        (1, 28),
        (2, 80),
    ]
    for ax, (index, ymax) in zip(axes.ravel()[:3], panels):
        _model_bars(
            ax,
            labels,
            [
                ("Keyword note", [row[index] for row in keyword], BLUE),
                ("After 12SL withdrawal", [row[index] for row in after], GREEN),
            ],
            "Records (%)",
            ymax,
        )
    statement = [
        [
            _pct(arms[name]["statement_mismatch"])
            for name in ("no_retrieval", "keyword", "check_12sl", "check_both")
        ]
        for arms in loaded
    ]
    names = ["No retrieval", "Keyword", "12SL withdrawal", "Both algorithms"]
    colors = [BLUE, BLUE2, GREEN, GREEN3]
    _model_bars(
        axes[1, 1],
        labels,
        [(name, [row[i] for row in statement], color) for i, (name, color) in enumerate(zip(names, colors))],
        "Statement mismatch (%)",
        46,
    )
    _style_quad(
        fig,
        axes,
        ["A  Expert diagnosis absent", "B  Uni-G rejects a criterion", "C  Release score", "D  Statement mismatch"],
    )
    save(fig, "fig_scores")
def bars_diagnosis():
    """Wide 2x2: class recall, false-diagnosis notes, and micro-F1."""
    labels, _loaded, prf = _cohort()
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    class_names = ["Normal", "Infarction", "ST/T", "Conduction", "Hypertrophy"]
    colors = [BLUE, BLUE2, GREEN, GREEN2, GREEN3]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    for ax, arm in (
        (axes[0, 0], "no_retrieval"),
        (axes[0, 1], "keyword"),
    ):
        series = []
        for name, code, color in zip(class_names, classes, colors):
            series.append(
                (
                    name,
                    [_pct(prf[item[2]][arm]["by_class"][code]["recall"]) for item in MODELS],
                    color,
                )
            )
        _model_bars(ax, labels, series, "Class recall (%)", 112)
    false_no = [_pct(prf[item[2]]["no_retrieval"]["notes_with_any_false_diagnosis"]) for item in MODELS]
    false_kw = [_pct(prf[item[2]]["keyword"]["notes_with_any_false_diagnosis"]) for item in MODELS]
    f1_no = [_pct(prf[item[2]]["no_retrieval"]["micro_f1"]) for item in MODELS]
    f1_kw = [_pct(prf[item[2]]["keyword"]["micro_f1"]) for item in MODELS]
    _model_bars(
        axes[1, 0],
        labels,
        [("No retrieval", false_no, BLUE), ("Keyword retrieval", false_kw, GREEN)],
        "False-diagnosis notes (%)",
        40,
    )
    _model_bars(
        axes[1, 1],
        labels,
        [("No retrieval", f1_no, BLUE), ("Keyword retrieval", f1_kw, GREEN)],
        "Micro-F1 (%)",
        100,
    )
    _style_quad(
        fig,
        axes,
        [
            "A  Class recall, no retrieval",
            "B  Class recall, keyword retrieval",
            "C  False-diagnosis notes",
            "D  Micro-F1",
        ],
    )
    save(fig, "fig_diagnosis")
def noise():
    labels = ["0.10", "0.25", "0.50"]
    series = [
        ("Baseline wander", [0.17, 0.50, 0.67]),
        ("Muscle artifact", [0.17, 0.50, 1.00]),
        ("Electrode motion", [0.33, 1.00, 2.50]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    grouped(ax, labels, series, "False criterion releases (%)", 3.3, [NAVY, TEAL, RED])
    ax.set_xlabel("Noise RMS relative to the electrocardiogram")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_noise")
def composite():
    fig = plt.figure(figsize=(7.35, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.25)
    \
\
    ax = fig.add_subplot(gs[0, 0])
    models = ["gpt-4.1-mini", "Qwen3-8B"]
    no_check = np.array([639, 880], dtype=float)
    false_pred = np.array([8, 211], dtype=float)
    total = no_check + false_pred
    p_no = 100 * no_check / total
    p_fp = 100 * false_pred / total
    x = np.arange(len(models))
    ax.bar(x, p_no, color="#D9DDE3", linewidth=0, label="No numeric check", zorder=3)
    ax.bar(x, p_fp, bottom=p_no, color=RED, linewidth=0, label="12SL predicate false", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of withdrawals (%)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    panel_letter(ax, "A")
    ax.set_title("What drove withdrawals", fontsize=8, pad=6, color=INK)
    \
\
    ax = fig.add_subplot(gs[0, 1])
    groups = ["Unmatched\nprompts", "Matched\nprompts"]
    no_rec = [77.2, 77.0]
    kw_rec = [54.6, 75.2]
    x = np.arange(len(groups))
    width = 0.32
    ax.bar(x - width / 2, no_rec, width=width, color=NAVY, linewidth=0, label="No retrieval", zorder=3)
    ax.bar(x + width / 2, kw_rec, width=width, color=TEAL, linewidth=0, label="Constrained retrieval", zorder=3)
    for xi, a, b in zip(x, no_rec, kw_rec):
        ax.text(xi - width / 2, a + 1.1, f"{a:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK)
        ax.text(xi + width / 2, b + 1.1, f"{b:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(45, 85)
    ax.set_ylabel("Diagnosis micro recall (%)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    panel_letter(ax, "B")
    ax.set_title("Prompt mismatch drove recall loss", fontsize=8, pad=6, color=INK)
    \
\
    ax = fig.add_subplot(gs[1, 0])
    items = ["Micro-F1", "Hypertrophy recall"]
    mean = [-0.9, -10.3]
    lo = [-1.4, -15.9]
    hi = [-0.4, -4.7]
    y = np.arange(len(items))[::-1]
    for yi, m, l, h in zip(y, mean, lo, hi):
        ci_dot(ax, yi, m, l, h, NAVY)
        ax.text(h + 0.35, yi, f"{m:.1f}", ha="left", va="center", fontsize=6.6, color=INK)
    ax.axvline(0, color="#8A93A3", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(items)
    ax.set_xlabel("Difference, constraint − no cards (percentage points)")
    ax.set_xlim(-18, 3)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)
    panel_letter(ax, "C")
    ax.set_title("Matched diagnosis deltas", fontsize=8, pad=6, color=INK)
    \
\
    ax = fig.add_subplot(gs[1, 1])
    labels = ["Numeric cards\nfree identifier", "Numeric cards\nschema identifier"]
    vals = [13.8, 3.0]
    ax.bar([0, 1], vals, color=[NAVY, TEAL], linewidth=0, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 18)
    ax.set_ylabel("Release score (%)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    panel_letter(ax, "D")
    ax.set_title("Schema identifier reduced format errors", fontsize=8, pad=6, color=INK)
    \
    fig.tight_layout()
    save(fig, "fig_composite")
def forest():
    path = os.path.join(PROJECT, "results", "ptbxl_baseline", "micro_f1_forest.json")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    rows = []
    for item in report.get("models", []):
        if "micro_f1" not in item:
            continue
        rows.append(
            {
                "label": item["label"],
                "n": item.get("scored"),
                "mean": 100 * float(item["micro_f1"]["difference"]),
                "lo": 100 * float(item["micro_f1"]["ci95"][0]),
                "hi": 100 * float(item["micro_f1"]["ci95"][1]),
            }
        )
    rows = sorted(rows, key=lambda r: r["mean"])
    labels = [f"{r['label']} (n={r['n']})" for r in rows]
    y = np.arange(len(rows))[::-1]
    \
    fig, ax = plt.subplots(figsize=(7.35, 3.8))
    for yi, r in zip(y, rows):
        ci_dot(ax, yi, r["mean"], r["lo"], r["hi"], NAVY)
    ax.axvline(0, color="#8A93A3", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Keyword − no retrieval micro-F1 (percentage points)")
    ax.set_xlim(-14, 10)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "fig_forest")
if __name__ == "__main__":
    workflow()
    endpoint()
    composite()
    statement()
    matched()
    bars_scores()
    bars_diagnosis()
    forest()
    noise()
    print("wrote figures")
