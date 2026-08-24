"""Day-5 statistical analysis — v2 after adversarial statistical review.

v1 flaws fixed here:
  - centering now uses positions 2-4 ONLY (v1 centered over 1-4, leaking the
    annotation-anchored, genre-confounded position-1 offset into every tested
    value as -delta/4);
  - inference is an EXACT SESSION-LABEL PERMUTATION TEST (8 candombe ensemble
    groups + 3 samba musician sets M2-M4 = C(11,3)=165 assignments) — v1's
    3-cluster percentile bootstrap had ~10 distinct resamples and was
    anticonservative; the same standard applies to every starred claim;
  - 2 orthogonal contrasts per register ("cliff" = p2 - (p3+p4)/2, "tail" =
    p3 - p4) replace 3 correlated per-position tests (centered 3 positions
    have 2 df); family-wise Holm correction over the 6 tests per family;
  - the primary tempo control is a permutation ANCOVA (genre effect with BPM
    covariate, labels permuted at session level); the BPM-overlap subset is
    demoted to a labeled sensitivity analysis with its subgenre composition
    reported (it shifts enredo from 23% to 51% of the samba side);
  - tempo correlations are re-run in MILLISECONDS to separate real tempo
    dependence from the fraction-of-beat units artifact;
  - subgenre inference is restricted to the tempo-overlapping pair
    (samba vs partido alto) with BPM covariate and within-session label
    permutation; the rest is explicitly tempo-entangled description;
    (v1 also had a dict-key collision silently dropping samba-vs-partido);
  - variance components via unbalanced method-of-moments (v1 compared raw
    SDs across aggregation levels).

Framing note for the paper (limitations): genre is perfectly confounded with
corpus (one corpus per genre); all claims are about candombe/samba AS
REPRESENTED BY THESE CORPORA. Day-6 must include a BPM-only baseline
classifier and a within-corpus session-discriminability control.

Outputs: results/stats_summary.csv, results/effect_sizes.png,
         results/subgenre_profiles.png, results/tempo_dependence.png.
"""

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(20260824)

REGISTER_PAIRS = [
    ("low", "piano", "surdo"),
    ("mid", "repique", "caixa_mid"),
    ("high", "chico", "tamborim"),
]
CONTRASTS = {
    "cliff": lambda d: d[2] - (d[3] + d[4]) / 2,   # early vs late half of beat
    "tail": lambda d: d[3] - d[4],                 # shape of the late half
}
N_PERM_STRAT = 4999


def per_track(df: pd.DataFrame, unit: str = "frac") -> pd.DataFrame:
    """Per track x band: positions 2-4 centered over 2-4 (+ raw pos1), meta.

    unit="frac": % of beat. unit="ms": milliseconds (per-track beat duration
    from mean_bpm) — used to separate tempo dependence from the units choice.
    """
    g = (
        df.groupby(["track_id", "band", "position"])
        .agg(deviation=("deviation", "mean"))
        .reset_index()
    )
    piv = g.pivot_table(index=["track_id", "band"], columns="position", values="deviation")
    meta = df.groupby("track_id").agg(
        genre=("genre", "first"), subgenre=("subgenre", "first"),
        session=("session", "first"), mean_bpm=("mean_bpm", "first"),
    )
    out = piv.reset_index().merge(meta, on="track_id")
    if unit == "ms":
        beat_ms = 60000.0 / out.mean_bpm
        for p in [1, 2, 3, 4]:
            out[p] = out[p] * beat_ms
    else:
        for p in [1, 2, 3, 4]:
            out[p] = out[p] * 100
    center = out[[2, 3, 4]].mean(axis=1)
    for p in [2, 3, 4]:
        out[p] = out[p] - center
    return out


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return 0.0 if sp == 0 else (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


def session_perm_p(values: pd.Series, sessions: pd.Series, genres: pd.Series,
                   bpm: pd.Series | None = None) -> tuple[float, float]:
    """Exact two-sided session-label permutation p for the genre effect.

    Statistic: difference of unweighted means of session means (bpm=None), or
    the genre coefficient of OLS value ~ genre + bpm (permutation ANCOVA).
    All C(11,3) genre-label assignments to sessions are enumerated.
    Returns (observed_stat, p).
    """
    data = {"v": np.asarray(values, float), "s": np.asarray(sessions),
            "g": np.asarray(genres)}
    if bpm is not None:
        data["bpm"] = np.asarray(bpm, float)
    df = pd.DataFrame(data).dropna()
    sess = sorted(df.s.unique())
    true_samba = frozenset(df.loc[df.g == "samba", "s"].unique())
    k = len(true_samba)

    def stat(samba_set):
        lab = df.s.isin(samba_set)
        if bpm is None:
            sm = df.groupby("s").v.mean()
            return sm[~sm.index.isin(samba_set)].mean() - sm[sm.index.isin(samba_set)].mean()
        x = np.column_stack([
            np.ones(len(df)), lab.to_numpy(float), df.bpm.to_numpy(float)
        ])
        beta, *_ = np.linalg.lstsq(x, df.v.to_numpy(float), rcond=None)
        return -beta[1]  # candombe minus samba

    obs = stat(true_samba)
    perms = [stat(frozenset(c)) for c in combinations(sess, k)]
    p = float(np.mean([abs(s) >= abs(obs) - 1e-12 for s in perms]))
    return float(obs), p


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    res = ROOT / "results"
    cand = pd.read_parquet(res / "deviations_candombe.parquet")
    brid = pd.read_parquet(res / "deviations_brid.parquet")
    brid = brid[brid.genre == "samba"]

    pt = {u: (per_track(cand, u), per_track(brid, u)) for u in ("frac", "ms")}
    pt_c, pt_b = pt["frac"]
    rows = []

    # ---------------- A. cross-genre contrasts (exact permutation) ----------
    print("=" * 70)
    print("A. CROSS-GENRE contrasts (centered over pos 2-4, % of beat)")
    print(f"   sessions: candombe={pt_c.session.nunique()} ensembles, "
          f"samba={pt_b.session.nunique()} musician sets (M2-M4) -> exact perm, "
          f"C(11,3)=165 assignments; Holm over 6 tests")
    a_tests = []
    for reg, band_c, band_s in REGISTER_PAIRS:
        both = pd.concat([pt_c[pt_c.band == band_c], pt_b[pt_b.band == band_s]])
        for cname, cf in CONTRASTS.items():
            v = cf(both)
            obs, p = session_perm_p(v, both.session, both.genre)
            g = hedges_g(cf(pt_c[pt_c.band == band_c]).dropna(),
                         cf(pt_b[pt_b.band == band_s]).dropna())
            a_tests.append(dict(section="cross_genre", register=reg, contrast=cname,
                                stat=obs, perm_p=p, hedges_g=g))
    for t, adj in zip(a_tests, holm([t["perm_p"] for t in a_tests])):
        t["holm_p"] = adj
        rows.append(t)
        print(f"  {t['register']:4s} {t['contrast']:5s}: diff={t['stat']:+.2f}%  "
              f"g={t['hedges_g']:+.2f}  perm p={t['perm_p']:.3f}  "
              f"Holm p={adj:.3f} {'*' if adj < 0.05 else 'ns'}")

    # ---------------- A2. primary tempo control: permutation ANCOVA ---------
    print("\nA2. TEMPO-CONTROLLED cross-genre (permutation ANCOVA: value ~ genre + BPM)")
    a2_tests = []
    for reg, band_c, band_s in REGISTER_PAIRS:
        both = pd.concat([pt_c[pt_c.band == band_c], pt_b[pt_b.band == band_s]])
        for cname, cf in CONTRASTS.items():
            v = cf(both)
            obs, p = session_perm_p(v, both.session, both.genre, bpm=both.mean_bpm)
            a2_tests.append(dict(section="cross_genre_ancova", register=reg,
                                 contrast=cname, stat=obs, perm_p=p))
    for t, adj in zip(a2_tests, holm([t["perm_p"] for t in a2_tests])):
        t["holm_p"] = adj
        rows.append(t)
        print(f"  {t['register']:4s} {t['contrast']:5s}: genre coef={t['stat']:+.2f}%  "
              f"perm p={t['perm_p']:.3f}  Holm p={adj:.3f} {'*' if adj < 0.05 else 'ns'}")

    # sensitivity (labeled, not confirmatory): BPM-overlap subset composition
    lo = max(pt_c.mean_bpm.min(), pt_b.mean_bpm.min())
    hi = min(pt_c.mean_bpm.max(), pt_b.mean_bpm.max())
    sub_b = pt_b[(pt_b.mean_bpm >= lo) & (pt_b.mean_bpm <= hi)].drop_duplicates("track_id")
    comp = sub_b.subgenre.value_counts(normalize=True).round(2).to_dict()
    print(f"\n  [sensitivity note] BPM overlap [{lo:.0f},{hi:.0f}]: samba subset "
          f"composition {comp} (full corpus: samba .46, partido .31, enredo .23) — "
          f"range-matching shifts populations; ANCOVA above is the primary control")

    # ---------------- C. tempo dependence: units check ----------------------
    print("\n" + "=" * 70)
    print("C. TEMPO DEPENDENCE (within samba): fraction-of-beat vs milliseconds")
    for cname, cf in CONTRASTS.items():
        for band in ["surdo", "caixa_mid", "tamborim"]:
            r = {}
            for u in ("frac", "ms"):
                b = pt[u][1]
                bdf = b[b.band == band]
                v = cf(bdf)
                ok = v.notna() & bdf.mean_bpm.notna()
                r[u] = stats.spearmanr(v[ok], bdf.mean_bpm[ok])
            rows.append(dict(section="tempo_units", register=band, contrast=cname,
                             rho_frac=r["frac"].statistic, p_frac=r["frac"].pvalue,
                             rho_ms=r["ms"].statistic, p_ms=r["ms"].pvalue))
            verdict = ("units artifact" if abs(r["frac"].statistic) > 0.4 > abs(r["ms"].statistic)
                       else "real tempo dependence" if abs(r["ms"].statistic) > 0.4
                       else "weak")
            print(f"  {band:9s} {cname:5s}: rho_frac={r['frac'].statistic:+.2f} "
                  f"rho_ms={r['ms'].statistic:+.2f} -> {verdict}")

    # ---------------- B. subgenres ------------------------------------------
    print("\n" + "=" * 70)
    print("B. SAMBA SUBGENRES")
    print("  Descriptive means (tempo-entangled: enredo 130 BPM vs samba 91 / partido 99):")
    for band in ["surdo", "caixa_mid", "tamborim"]:
        bdf = pt_b[pt_b.band == band]
        desc = bdf.groupby("subgenre")[[2, 3, 4]].mean().round(2)
        print(f"  [{band}]")
        print(desc.to_string(header=[f"pos{p}" for p in [2, 3, 4]]))
    print("\n  Inference (only tempo-overlapping pair: samba vs partido alto, "
          "BPM covariate, subgenre labels permuted WITHIN sessions):")
    pair = pt_b[pt_b.subgenre.isin(["samba", "partido alto"])]
    b_tests = []
    for band in ["surdo", "caixa_mid", "tamborim"]:
        bdf = pair[pair.band == band].copy()
        for cname, cf in CONTRASTS.items():
            v = cf(bdf)
            ok = v.notna() & bdf.mean_bpm.notna()
            d, s, gsub, bpm = v[ok], bdf.session[ok], bdf.subgenre[ok], bdf.mean_bpm[ok]

            def coef(labels):
                x = np.column_stack([
                    np.ones(len(d)), (labels == "partido alto").to_numpy(float),
                    bpm.to_numpy(float),
                ])
                beta, *_ = np.linalg.lstsq(x, d.to_numpy(float), rcond=None)
                return beta[1]

            obs = coef(gsub)
            perms = []
            for _ in range(N_PERM_STRAT):
                lab = gsub.copy()
                for sess in lab.groupby(s).groups.values():
                    lab.loc[sess] = rng.permutation(lab.loc[sess].to_numpy())
                perms.append(coef(lab))
            p = float((np.sum(np.abs(perms) >= abs(obs) - 1e-12) + 1) / (len(perms) + 1))
            b_tests.append(dict(section="subgenre_pa_vs_sa", register=band,
                                contrast=cname, stat=obs, perm_p=p))
    for t, adj in zip(b_tests, holm([t["perm_p"] for t in b_tests])):
        t["holm_p"] = adj
        rows.append(t)
        print(f"  {t['register']:9s} {t['contrast']:5s}: coef={t['stat']:+.2f}%  "
              f"perm p={t['perm_p']:.4f}  Holm p={adj:.4f} {'*' if adj < 0.05 else 'ns'}")

    # ---------------- D. variance components (method of moments) ------------
    print("\n" + "=" * 70)
    print("D. VARIANCE COMPONENTS (MoM, centered %, pos 2-4 pooled)")
    for name, ptx, raw in [("candombe", pt_c, cand), ("samba", pt_b, brid)]:
        for band in sorted(ptx.band.unique()):
            bdf = ptx[ptx.band == band]
            long = bdf.melt(id_vars=["track_id", "session"], value_vars=[2, 3, 4],
                            var_name="position", value_name="dev").dropna()
            comps = {"sigma_session": [], "sigma_track": []}
            for pos, sub in long.groupby("position"):
                var_track = sub.groupby("session").dev.var().mean()
                n_bar = sub.groupby("session").size().mean()
                var_sess_means = sub.groupby("session").dev.mean().var()
                sigma2_sess = max(0.0, var_sess_means - var_track / n_bar)
                comps["sigma_session"].append(np.sqrt(sigma2_sess))
                comps["sigma_track"].append(np.sqrt(var_track))
            within_track = (
                raw[raw.band == band].groupby(["track_id", "position"])
                .deviation.std().mean() * 100
            )
            ss, st = np.mean(comps["sigma_session"]), np.mean(comps["sigma_track"])
            rows.append(dict(section="variance", register=f"{name}:{band}",
                             sigma_session=ss, sigma_track=st,
                             sigma_onset=within_track))
            print(f"  {name:9s} {band:9s}: sigma_session={ss:.2f}  "
                  f"sigma_track={st:.2f}  sigma_onset={within_track:.2f}")

    pd.DataFrame(rows).to_csv(res / "stats_summary.csv", index=False)
    print(f"\n-> {res / 'stats_summary.csv'}")

    # ---------------- figures ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    er = [r for r in rows if r["section"] == "cross_genre"]
    ea = {(r["register"], r["contrast"]): r for r in rows
          if r["section"] == "cross_genre_ancova"}
    x = np.arange(len(er))
    # color = the PRIMARY inference (tempo-controlled permutation ANCOVA)
    colors = [
        "#14675A" if ea[(r["register"], r["contrast"])]["holm_p"] < 0.05 else "#B9C2B4"
        for r in er
    ]
    ax.bar(x, [r["hedges_g"] for r in er], color=colors, width=0.6)
    for i, r in enumerate(er):
        a = ea[(r["register"], r["contrast"])]
        offset = 0.07 if r["hedges_g"] >= 0 else -0.19
        ax.text(i, r["hedges_g"] + offset,
                f"ANCOVA p={a['holm_p']:.2f}", ha="center", fontsize=7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['register']}\n{r['contrast']}" for r in er], fontsize=8)
    ax.set_ylabel("Hedges' g (candombe - samba)")
    ax.set_ylim(min(r["hedges_g"] for r in er) - 0.45,
                max(r["hedges_g"] for r in er) + 0.45)
    ax.set_title(
        "Cross-genre timing contrasts — dark: significant under\n"
        "tempo-controlled session-permutation ANCOVA (Holm)", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(res / "effect_sizes.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, band in zip(axes, ["surdo", "caixa_mid", "tamborim"]):
        for s in ["samba", "samba de enredo", "partido alto"]:
            bdf = pt_b[(pt_b.band == band) & (pt_b.subgenre == s)]
            means = [bdf[p].mean() for p in [2, 3, 4]]
            sems = [bdf[p].sem() * stats.t.ppf(0.975, max(bdf[p].count() - 1, 1))
                    for p in [2, 3, 4]]
            ax.errorbar([2, 3, 4], means, yerr=sems, marker="o", capsize=2, label=s)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(band, fontsize=10)
        ax.set_xticks([2, 3, 4])
    axes[0].set_ylabel("centered deviation (% of beat)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Samba subgenre profiles (descriptive; subgenre and tempo entangled)")
    fig.tight_layout()
    fig.savefig(res / "subgenre_profiles.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, u, unit_lbl in [(axes[0], "frac", "% of beat"), (axes[1], "ms", "ms")]:
        b = pt[u][1]
        bdf = b[b.band == "caixa_mid"]
        v = CONTRASTS["cliff"](bdf)
        ax.scatter(bdf.mean_bpm, v, s=16, alpha=0.7)
        ax.set_xlabel("track mean BPM")
        ax.set_ylabel(f"cliff contrast ({unit_lbl})")
        ax.axhline(0, color="gray", lw=0.8)
    fig.suptitle("Tempo dependence of samba mid-register cliff: units comparison")
    fig.tight_layout()
    fig.savefig(res / "tempo_dependence.png", dpi=150)
    print("figures -> effect_sizes.png, subgenre_profiles.png, tempo_dependence.png")


if __name__ == "__main__":
    main()
