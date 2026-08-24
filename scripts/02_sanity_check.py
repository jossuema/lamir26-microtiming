"""Sanity checks for the deviation pipeline against published findings.

Checks (qualitative, against the literature):
  1. Deviations are systematic, not noise: per-position means differ from 0
     with tight confidence intervals, consistently across recordings.
  2. Samba (BRID): anticipation of the 3rd and 4th sixteenth notes
     (Naveda et al. 2011) -> mean deviation < 0 at positions 3 and 4.
  3. Candombe chico vs samba timekeeper profiles separate (Fuentes et al.
     2019, Fig. 6 showed visible separation for chico vs tamborim).

Statistics: CIs are computed across TRACKS (per-track position means first,
then a t-based CI over the n_tracks means). Pooling all onsets and using
SEM would be pseudoreplication — onsets within a recording are clustered.

Position 1 caveat: the grid is anchored at the annotated beat, and beat
annotations tend to sit on played onsets of a reference instrument — so
position-1 deviations are partly relative to annotation convention, which
differs between corpora and is confounded with genre. Cross-genre claims
(and classifier features) should rest on positions 2-4.

Outputs: results/sanity_profiles.png + printed per-position stats.
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


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-position mean deviation with a cluster-aware 95% CI (% of beat).

    Aggregates to per-track position means first; the CI is t-based across
    tracks, so n = number of recordings, not number of onsets.

    Two views of the same profile:
      raw (mean_pct)          — includes a constant per-track offset from
                                detector latency + annotation convention;
      centered (centered_pct) — each track's position means minus that
                                track's cross-position mean. The constant
                                offset cancels, leaving profile SHAPE; this
                                is the view comparable to the literature's
                                "anticipation" claims until per-band latency
                                is calibrated on solo tracks.
    """
    per_track = (
        df.groupby(["track_id", "position"])["deviation"].mean().unstack()
    )
    centered = per_track.sub(per_track.mean(axis=1), axis=0)

    def across_tracks(pt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        mean = pt.mean()
        n = pt.count()
        tcrit = pd.Series({p: stats.t.ppf(0.975, max(k - 1, 1)) for p, k in n.items()})
        ci = tcrit * pt.sem()
        return mean, ci

    mean_raw, ci_raw = across_tracks(per_track)
    mean_c, ci_c = across_tracks(centered)
    out = pd.DataFrame(
        {
            "mean_pct": mean_raw * 100,
            "ci95_pct": ci_raw * 100,
            "centered_pct": mean_c * 100,
            "centered_ci95": ci_c * 100,
            "n_tracks": per_track.count(),
            "n_onsets": df.groupby("position").size(),
        }
    )
    out.index.name = "position"
    return out


def main() -> None:
    res = ROOT / "results"
    cand = pd.read_parquet(res / "deviations_candombe.parquet")
    brid_path = res / "deviations_brid.parquet"
    brid_all = pd.read_parquet(brid_path) if brid_path.exists() else None
    brid = None
    if brid_all is not None:
        brid = brid_all[brid_all.genre == "samba"]
        excluded = brid_all[brid_all.genre != "samba"]
        if not excluded.empty:
            counts = excluded.groupby("subgenre")["track_id"].nunique().to_dict()
            print(f"BRID: excluded non-samba rhythms from profiles: {counts}")

    print("=" * 60)
    print("CANDOMBE per-band profiles (mean deviation, % of beat)")
    print("(CIs across tracks; position 1 is annotation-anchored — see header)")
    for band in ["chico", "repique", "piano", "full"]:
        sub = cand[cand.band == band]
        if sub.empty:
            continue
        print(f"\n[{band}] n_onsets={len(sub)}, n_tracks={sub.track_id.nunique()}")
        print(profile(sub).round(2))

    # consistency across recordings: per-track profile SD at each position
    piv = (
        cand[cand.band == "chico"]
        .groupby(["track_id", "position"])["deviation"]
        .mean()
        .unstack()
    )
    print("\nchico: across-track SD of per-track position means (% of beat):")
    print((piv.std() * 100).round(2))

    if brid is not None and not brid.empty:
        print("\n" + "=" * 60)
        print("BRID (samba-family mixtures) per-band profiles")
        for band in ["tamborim", "caixa_mid", "surdo", "full"]:
            sub = brid[brid.band == band]
            if sub.empty:
                continue
            print(f"\n[{band}] n_onsets={len(sub)}, n_tracks={sub.track_id.nunique()}")
            print(profile(sub).round(2))
        t34 = profile(brid[brid.band == "tamborim"])
        if {3, 4}.issubset(t34.index):
            # Naveda et al.'s anticipation is a SHAPE claim; raw deviations
            # carry a constant detector-latency offset, so evaluate centered.
            anticip = (t34.loc[3, "centered_pct"] < 0) and (t34.loc[4, "centered_pct"] < 0)
            sig = (
                t34.loc[3, "centered_pct"] + t34.loc[3, "centered_ci95"] < 0
                and t34.loc[4, "centered_pct"] + t34.loc[4, "centered_ci95"] < 0
            )
            print(
                f"\nNaveda et al. check — tamborim positions 3 & 4 anticipated "
                f"(centered profile): {'PASS' if anticip else 'FAIL'}"
                f"{' (CI excludes 0)' if sig else ''} "
                f"(pos3={t34.loc[3, 'centered_pct']:.2f}%, pos4={t34.loc[4, 'centered_pct']:.2f}%)"
            )
        else:
            print("\nNaveda et al. check: SKIPPED (positions 3/4 missing in tamborim band)")

    # figure: raw (left, offset included) vs centered shape (right)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    plots = [("candombe " + b, cand[cand.band == b]) for b in ["chico", "repique", "piano"]]
    if brid is not None:
        plots.append(("samba tamborim", brid[brid.band == "tamborim"]))
        plots.append(("samba surdo", brid[brid.band == "surdo"]))
    for label, sub in plots:
        if sub.empty:
            continue
        p = profile(sub)
        ls = "--" if label.startswith("samba") else "-"
        axes[0].errorbar(
            p.index, p.mean_pct, yerr=p.ci95_pct,
            marker="o", capsize=3, label=label, ls=ls,
        )
        axes[1].errorbar(
            p.index, p.centered_pct, yerr=p.centered_ci95,
            marker="o", capsize=3, label=label, ls=ls,
        )
    for ax, title in zip(
        axes,
        ["raw (includes detector-latency offset)", "centered per track (profile shape)"],
    ):
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("sixteenth-note position within beat")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("mean deviation (% of beat)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Microtiming profiles: candombe roles vs samba timekeepers")
    fig.tight_layout()
    out = res / "sanity_profiles.png"
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
