import json
import os
\
import matplotlib
\
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
\
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }
)
\
NAVY = "#3C5488"
TEAL = "#00A087"
RED = "#E64B35"
INK = "#222222"
SLATE = "#5D6D7E"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "Figs")
\
MODELS = [
    ("gpt-4.1-mini", "gpt_heldout"),
    ("Qwen3-8B", "qwen3_8b"),
    ("Qwen3-14B", "qwen3_14b"),
    ("Gemma 3 12B", "gemma3_12b"),
    ("MedGemma 4B", "medgemma_4b"),
    ("GLM-5.3-Flash", "glm_heldout"),
]
RADAR_COLORS = ["#3C5488", "#00A087", "#4DBBD5", "#E64B35", "#F39B7F", "#8491B4"]
PERF = LinearSegmentedColormap.from_list("perf", ["#F4F7FB", "#9BB6D4", "#3C5488"])
HARM = LinearSegmentedColormap.from_list("harm", ["#FDEDEC", "#F5B7B1", "#E64B35"])
\
\
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
def load_prf():
    path = os.path.join(PROJECT, "results", "ptbxl_baseline", "diagnosis_prf.json")
    with open(path, encoding="utf-8") as handle:
        arms = json.load(handle)["arms"]
    rows = []
    for label, key in MODELS:
        block = arms[key]
        rows.append(
            {
                "label": f"{label} ({block['scored']:,})",
                "short": label,
                "n": block["scored"],
                "no": metrics(block["no_retrieval"]),
                "kw": metrics(block["keyword"]),
            }
        )
    return rows
def metrics(arm):
    return np.array(
        [
            arm["micro_precision"],
            arm["micro_recall"],
            arm["micro_f1"],
            arm["notes_with_any_false_diagnosis"],
        ],
        dtype=float,
    )
def ink_for(rgba):
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.58 else INK
def heatmap(rows):
    labels = [row["label"] for row in rows]
    columns = ["Precision", "Recall", "F1", "False\ndiagnosis"]
    mats = [np.vstack([row["no"] for row in rows]), np.vstack([row["kw"] for row in rows])]
    both = np.vstack(mats)
    perf_lo, perf_hi = both[:, :3].min() - 0.02, both[:, :3].max() + 0.02
    harm_lo, harm_hi = both[:, 3].min() - 0.015, both[:, 3].max() + 0.015
    \
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.55), sharey=True)
    titles = ["A   No retrieval", "B   Keyword retrieval"]
    for ax, mat, title in zip(axes, mats, titles):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, len(rows))
        ax.set_aspect("equal")
        for i in range(len(rows)):
            for j in range(4):
                value = mat[i, j]
                if j < 3:
                    tone = np.clip((value - perf_lo) / (perf_hi - perf_lo), 0, 1)
                    color = PERF(tone)
                else:
                    tone = np.clip((value - harm_lo) / (harm_hi - harm_lo), 0, 1)
                    color = HARM(tone)
                ax.add_patch(
                    Rectangle((j, len(rows) - 1 - i), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
                )
                ax.text(
                    j + 0.5,
                    len(rows) - 1 - i + 0.5,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=ink_for(color),
                )
        ax.set_xticks(np.arange(4) + 0.5)
        ax.set_xticklabels(columns)
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels(labels[::-1])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=8)
    cax1 = fig.add_axes([0.18, -0.02, 0.28, 0.035])
    cax2 = fig.add_axes([0.58, -0.02, 0.28, 0.035])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=PERF, norm=plt.Normalize(100 * perf_lo, 100 * perf_hi)),
        cax=cax1,
        orientation="horizontal",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=HARM, norm=plt.Normalize(100 * harm_lo, 100 * harm_hi)),
        cax=cax2,
        orientation="horizontal",
    )
    cax1.set_title("Precision, recall, and F1 (%)", fontsize=7, color=INK, loc="left", pad=2)
    cax2.set_title("False-diagnosis notes (%)", fontsize=7, color=INK, loc="left", pad=2)
    for cax in (cax1, cax2):
        cax.tick_params(labelsize=6.5, length=2, width=0.4)
    fig.subplots_adjust(wspace=0.08, bottom=0.18)
    save(fig, "fig_model_heatmap")
def ribbon(ax, x0, x1, y0, h0, y1, h1, color):
    t = np.linspace(0, 1, 48)
    s = t * t * (3 - 2 * t)
    xs = x0 + (x1 - x0) * t
    top = (y0 + h0) + ((y1 + h1) - (y0 + h0)) * s
    bot = y0 + (y1 - y0) * s
    ax.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([bot, top[::-1]]), color=color, alpha=0.55, lw=0)
def node(ax, x, y, w, h, color):
    if h <= 0:
        return
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", linewidth=0))
def sankey_panel(ax, title, applied, unchecked, failed):
    retained = applied - unchecked - failed
    numeric = failed + retained
    gap = applied * 0.04
    node_w = 0.11
    xs = [0.0, 1.05, 2.10]
    span = applied + 2 * gap
    \
    left_kept = (span - applied) / 2
    left_failed = left_kept + retained
    left_unchecked = left_failed + failed
    mid_kept = left_kept
    mid_failed = left_failed
    mid_unchecked = left_unchecked + gap
    right_kept = left_kept
    right_failed = left_failed
    right_unchecked = left_unchecked + 2 * gap
    \
    ribbon(ax, xs[0] + node_w, xs[1], left_unchecked, unchecked, mid_unchecked, unchecked, SLATE)
    ribbon(ax, xs[0] + node_w, xs[1], left_failed, failed, mid_failed, failed, RED)
    ribbon(ax, xs[0] + node_w, xs[1], left_kept, retained, mid_kept, retained, TEAL)
    ribbon(ax, xs[1] + node_w, xs[2], mid_unchecked, unchecked, right_unchecked, unchecked, SLATE)
    ribbon(ax, xs[1] + node_w, xs[2], mid_failed, failed, right_failed, failed, RED)
    ribbon(ax, xs[1] + node_w, xs[2], mid_kept, retained, right_kept, retained, TEAL)
    \
    node(ax, xs[0], left_kept, node_w, applied, NAVY)
    node(ax, xs[1], mid_unchecked, node_w, unchecked, SLATE)
    node(ax, xs[1], mid_failed, node_w, failed, RED)
    node(ax, xs[1], mid_kept, node_w, retained, TEAL)
    node(ax, xs[2], right_unchecked, node_w, unchecked, SLATE)
    node(ax, xs[2], right_failed, node_w, failed, RED)
    node(ax, xs[2], right_kept, node_w, retained, TEAL)
    \
    def side(x, y, h, text, align):
        ax.text(x, y + max(h, applied * 0.02) / 2, text, ha=align, va="center", fontsize=6.7, color=INK, linespacing=1.15)
    side(xs[0] - 0.05, left_kept, applied, f"Applied\n{applied:,}", "right")
    side(xs[2] + node_w + 0.05, right_unchecked, unchecked, f"Withdrawn, no check\n{unchecked:,}", "left")
    side(xs[2] + node_w + 0.05, right_failed, failed, f"Withdrawn, 12SL false\n{failed:,}", "left")
    side(xs[2] + node_w + 0.05, right_kept, retained, f"Retained\n{retained:,}", "left")
    \
    ax.set_xlim(-0.78, 3.85)
    ax.set_ylim(-span * 0.16, span * 1.02)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=9, color=INK, pad=4)
    ax.text(xs[0] + node_w / 2, -span * 0.07, "Applied", ha="center", va="top", fontsize=7, color="#4B5563")
    ax.text(xs[1] + node_w / 2, -span * 0.07, "12SL check", ha="center", va="top", fontsize=7, color="#4B5563")
    ax.text(xs[2] + node_w / 2, -span * 0.07, "Outcome", ha="center", va="top", fontsize=7, color="#4B5563")
def sankey():
    path = os.path.join(PROJECT, "results", "ptbxl_baseline", "retention_paired.json")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.85))
    specs = [
        ("A   gpt-4.1-mini", report["gpt-4.1-mini"]),
        ("B   Qwen3-8B", report["Qwen3-8B"]),
    ]
    for ax, (title, block) in zip(axes, specs):
        sankey_panel(ax, f"{title}   n = {block['patients']:,}", block["applied"], block["no_check"], block["predicate_false"])
    fig.tight_layout()
    save(fig, "fig_verification_sankey")
def radar(rows):
    names = ["Precision", "Recall", "F1", "No false\ndiagnosis"]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.9), subplot_kw={"projection": "polar"})
    titles = ["A   No retrieval", "B   Keyword retrieval"]
    keys = ["no", "kw"]
    for ax, title, key in zip(axes, titles, keys):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), names, fontsize=7.5)
        ax.set_ylim(55, 92)
        ax.set_yticks([60, 70, 80, 90])
        ax.set_yticklabels(["60", "70", "80", "90"], fontsize=6, color="#6B7280")
        ax.tick_params(axis="x", pad=6)
        ax.spines["polar"].set_color("#D5DBE3")
        ax.grid(color="#E6E9EE", lw=0.6)
        for row, color in zip(rows, RADAR_COLORS):
            values = row[key].copy()
            values[3] = 1 - values[3]
            pct = np.concatenate([100 * values, [100 * values[0]]])
            ax.plot(angles, pct, color=color, lw=1.45, label=row["short"])
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    save(fig, "fig_model_radar")
if __name__ == "__main__":
    cohort = load_prf()
    heatmap(cohort)
    sankey()
    radar(cohort)
