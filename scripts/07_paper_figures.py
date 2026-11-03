"""Paper-ready figures (vector PDF, no suptitles, register labels, Okabe-Ito).

Differences from the working figures in results/:
  - no baked-in suptitles (LaTeX captions carry that content);
  - Fig 1 legend uses REGISTER terms (the paper's discipline), instrument
    names in parentheses; centered panel uses the ANALYSIS centering
    (positions 2-4) so text and figure quote the same numbers; raw panel
    keeps positions 1-4 for completeness;
  - colorblind-safe Okabe-Ito palette; fonts sized for \\columnwidth;
  - ablation panel axis states the sign convention explicitly.

Outputs: paper/figs/fig_profiles.pdf, fig_effects.pdf, fig_classifier.pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7})

OKABE = {"high_c": "#0072B2", "mid_c": "#E69F00", "low_c": "#009E73",
         "high_s": "#D55E00", "low_s": "#CC79A7"}
FIGS = ROOT / "paper/figs"
RES = ROOT / "results"


def per_track_positions(df, band, positions, center_positions):
    sub = df[(df.band == band) & (df.position.isin(positions))]
    pt = sub.groupby(["track_id", "position"]).deviation.mean().unstack()
    c = sub.groupby(["track_id", "position"]).deviation.mean().unstack()
    c = c[center_positions].sub(c[center_positions].mean(axis=1), axis=0) * 100
    raw = pt * 100
    def agg(x):
        m = x.mean()
        ci = x.sem() * stats.t.ppf(0.975, x.count().clip(lower=2) - 1)
        return m, ci
    return agg(raw), agg(c)


def fig_profiles(cand, brid):
    series = [
        (r"candombe high ($\it{chico}$)", cand, "chico", OKABE["high_c"], "-"),
        (r"candombe mid ($\it{repique}$)", cand, "repique", OKABE["mid_c"], "-"),
        (r"candombe low ($\it{piano}$)", cand, "piano", OKABE["low_c"], "-"),
        (r"samba high ($\it{tamborim}$ range)", brid, "tamborim", OKABE["high_s"], "--"),
        (r"samba low ($\it{surdo}$ range)", brid, "surdo", OKABE["low_s"], "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for label, df, band, color, ls in series:
        (rm, rci), (cm, cci) = per_track_positions(df, band, [1, 2, 3, 4], [2, 3, 4])
        axes[0].errorbar(rm.index, rm, yerr=rci, marker="o", ms=3, capsize=2,
                         label=label, color=color, ls=ls, lw=1.2)
        axes[1].errorbar(cm.index, cm, yerr=cci, marker="o", ms=3, capsize=2,
                         label=label, color=color, ls=ls, lw=1.2)
    for ax, xt in [(axes[0], [1, 2, 3, 4]), (axes[1], [2, 3, 4])]:
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xticks(xt)
        ax.set_xlabel("sixteenth-note position")
    axes[0].set_ylabel("deviation (% of beat)")
    axes[0].set_title("raw (positions 1–4)")
    axes[1].set_title("centered over positions 2–4")
    axes[1].legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_profiles.pdf")


def fig_effects():
    s = pd.read_csv(RES / "stats_summary.csv")
    er = s[s.section == "cross_genre"].reset_index()
    ea = s[s.section == "cross_genre_ancova"].set_index(["register", "contrast"])
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(er))
    colors, labels = [], []
    for r in er.itertuples():
        core = bool(ea.loc[(r.register, r.contrast), "robust_core"])
        colors.append("#14675A" if core else "#C4CBBF")
        labels.append(f"{r.register}\n{r.contrast}")
    ax.bar(x, er.hedges_g, color=colors, width=0.62)
    for i, r in enumerate(er.itertuples()):
        hp = ea.loc[(r.register, r.contrast), "holm_cr1"]
        off = 0.08 if r.hedges_g >= 0 else -0.22
        ax.text(i, r.hedges_g + off, f"{hp:.2f}", ha="center", fontsize=7)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Hedges' g (candombe − samba)")
    ax.set_ylim(er.hedges_g.min() - 0.5, er.hedges_g.max() + 0.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_effects.pdf")


def fig_classifier():
    c = pd.read_csv(RES / "classifier_summary.csv")
    cc = c[c["sample"] == "complete-case"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    names = ["timing", "BPM", "timing+BPM"]
    x = np.arange(3)
    axes[0].bar(x - 0.19, [cc.loc[n, "lopo_bal"] for n in names], width=0.38,
                label="balanced accuracy", color="#14675A")
    axes[0].bar(x + 0.19, [cc.loc[n, "lopo_auc"] for n in names], width=0.38,
                label="mean per-fold AUC", color="#A5641C")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower right")
    axes[0].set_title("leave-one-pair-out CV")
    regs = ["low", "mid", "high"]
    drops = [cc.loc[f"ablation drop-{r}", "stat"] for r in regs]
    axes[1].bar(regs, drops, color="#14675A", width=0.5)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_ylabel("accuracy drop when removed")
    axes[1].set_title("register ablation (positive = necessary)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_classifier.pdf")


if __name__ == "__main__":
    cand = pd.read_parquet(RES / "deviations_candombe.parquet")
    brid = pd.read_parquet(RES / "deviations_brid.parquet")
    brid = brid[brid.genre == "samba"]
    fig_profiles(cand, brid)
    fig_effects()
    fig_classifier()
    print("-> paper/figs/fig_profiles.pdf, fig_effects.pdf, fig_classifier.pdf")
