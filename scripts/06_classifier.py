"""Day-6 discriminability experiment — v2 after adversarial ML review.

v1 flaws fixed here:
  - features are now the 6 IDENTIFIABLE contrasts (cliff = p2-(p3+p4)/2,
    tail = p3-p4, per low/mid/high register; day-5 basis). v1's 12 centered
    positions had rank 8 (each triplet sums to 0), so per-feature importances
    were allocated by the L2 penalty, not the data. The near-collinear "full"
    band is dropped from features.
  - PRIMARY sample = complete cases (all 22 tracks with imputed cells are
    samba — missingness was a perfect one-directional genre cue). Sensitivity
    runs: full data with imputation; missingness-indicators-only; no-high-
    register.
  - descriptive metrics use LEAVE-ONE-PAIR-OUT CV (8x3=24 folds holding out
    one candombe + one samba session: both classes per fold, per-fold AUC
    valid). v1 pooled AUC over single-class LOGO folds compared scores from
    11 differently calibrated models — invalid, and it reversed the
    timing-vs-BPM ranking.
  - INFERENCE stays the exact session permutation of pooled LOGO balanced
    accuracy over all C(11,3)=165 assignments (immune to the calibration
    artifact: null and observed share it), now ALSO applied to the
    complementarity DIFFERENCE bal_acc(timing+BPM) - bal_acc(BPM-only).
  - session control: v1's 8-way leave-one-track-out had classes with n=1
    (recall 0 by construction). v2 reports (a) the top null partitions of the
    165 (do corpus-mixing partitions beat the true labeling?) and (b) mean
    pairwise 2-way session discrimination within corpus (sessions with >=4
    tracks) as the commensurable statistic.

Claim wording (README): the CORPORA are discriminable from timing shape
alone; genre is perfectly confounded with corpus, so genre attribution is
conditional — state alternative measurement-route explanations.

Outputs: results/classifier_summary.csv, results/classifier_importances.png,
         results/classifier_accuracy.png.
"""

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from microtiming.features import build_features

RNG_SEED = 20260825
REGS = ["low", "mid", "high"]


def make_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000),
    )


def make_rf():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=RNG_SEED, min_samples_leaf=2),
    )


def contrast_features(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = feats.copy()
    cols = []
    for reg in REGS:
        out[f"c_{reg}_cliff"] = out[f"f_{reg}_p2"] - (
            out[f"f_{reg}_p3"] + out[f"f_{reg}_p4"]) / 2
        out[f"c_{reg}_tail"] = out[f"f_{reg}_p3"] - out[f"f_{reg}_p4"]
        cols += [f"c_{reg}_cliff", f"c_{reg}_tail"]
    return out, cols


def logo_pooled_bal(X, y, groups, model_fn):
    """Pooled balanced accuracy under leave-one-group-out (single-class folds
    are fine for POOLED bal_acc: it decomposes into per-class recalls)."""
    y_true, y_pred = [], []
    for gname in np.unique(groups):
        test = groups == gname
        m = model_fn()
        m.fit(X[~test], y[~test])
        y_true.extend(y[test]); y_pred.extend(m.predict(X[test]))
    return balanced_accuracy_score(y_true, y_pred)


def lopo_eval(X, y, groups, genre_of_group, model_fn):
    """Leave-one-pair-out: hold out one candombe + one samba session per fold
    (both classes present -> per-fold AUC valid). Returns pooled bal_acc,
    mean per-fold AUC, per-track correctness (for paired tests)."""
    cand_g = [g for g, gen in genre_of_group.items() if gen == "candombe"]
    samba_g = [g for g, gen in genre_of_group.items() if gen == "samba"]
    y_true, y_pred, aucs = [], [], []
    correct = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    for gc in cand_g:
        for gs in samba_g:
            test = (groups == gc) | (groups == gs)
            m = model_fn()
            m.fit(X[~test], y[~test])
            pred = m.predict(X[test])
            proba = m.predict_proba(X[test])[:, 1]
            y_true.extend(y[test]); y_pred.extend(pred)
            aucs.append(roc_auc_score(y[test], proba))
            correct[test] += (pred == y[test])
            counts[test] += 1
    pooled_bal = balanced_accuracy_score(y_true, y_pred)
    track_acc = np.divide(correct, counts, out=np.zeros_like(correct),
                          where=counts > 0)
    return pooled_bal, float(np.mean(aucs)), track_acc


def exact_perm(X, feats, sessions, model_fn, stat_fn):
    """Evaluate stat_fn over all C(11,3) genre assignments to sessions.
    stat_fn(y_perm) -> float. Returns (observed, p, all_scores_with_sets)."""
    sess_genre = feats.groupby("session").genre.first()
    true_samba = frozenset(sess_genre[sess_genre == "samba"].index)
    results = []
    for samba_set in combinations(sorted(sessions), len(true_samba)):
        y_perm = (~feats.session.isin(samba_set)).to_numpy().astype(int)
        if len(np.unique(y_perm)) < 2:
            continue
        results.append((frozenset(samba_set), stat_fn(y_perm)))
    obs = next(s for ss, s in results if ss == true_samba)
    p = float(np.mean([s >= obs - 1e-12 for _, s in results]))
    return obs, p, results


def main() -> None:
    res = ROOT / "results"
    cand = pd.read_parquet(res / "deviations_candombe.parquet")
    brid = pd.read_parquet(res / "deviations_brid.parquet")
    brid = brid[brid.genre == "samba"]

    raw = build_features([cand, brid])
    raw, ccols = contrast_features(raw)
    full = raw
    cc = raw[raw.n_missing == 0].reset_index(drop=True)
    print(f"full data: {len(full)} tracks | complete-case (PRIMARY): {len(cc)} "
          f"({int((full.n_missing > 0).sum())} excluded, all "
          f"{full[full.n_missing > 0].genre.unique().tolist()})")

    rows = []

    def run_suite(feats, label):
        X_t = feats[ccols].to_numpy()
        X_b = feats[["mean_bpm"]].to_numpy()
        X_tb = feats[ccols + ["mean_bpm"]].to_numpy()
        y = (feats.genre == "candombe").to_numpy().astype(int)
        groups = feats.session.to_numpy()
        genre_of_group = feats.groupby("session").genre.first().to_dict()
        sessions = list(genre_of_group)

        print(f"\n--- {label} ---")
        out = {}
        for name, X in [("timing", X_t), ("BPM", X_b), ("timing+BPM", X_tb)]:
            bal, auc, track_acc = lopo_eval(X, y, groups, genre_of_group, make_logistic)
            out[name] = dict(bal=bal, auc=auc, track_acc=track_acc)
            print(f"  {name:10s} (logistic): LOPO pooled bal_acc={bal:.3f}  "
                  f"mean per-fold AUC={auc:.3f}")
            rows.append(dict(sample=label, model=name, lopo_bal=bal, lopo_auc=auc))
        bal_rf, auc_rf, _ = lopo_eval(X_t, y, groups, genre_of_group, make_rf)
        print(f"  timing (RF):          LOPO pooled bal_acc={bal_rf:.3f}  "
              f"mean per-fold AUC={auc_rf:.3f}")
        rows.append(dict(sample=label, model="timing (RF)", lopo_bal=bal_rf,
                         lopo_auc=auc_rf))
        return feats, X_t, X_b, X_tb, y, groups, sessions, out

    # PRIMARY: complete case
    feats, X_t, X_b, X_tb, y, groups, sessions, lopo = run_suite(cc, "complete-case")

    # exact permutations (LOGO pooled bal_acc; identical statistic under null)
    print("\nEXACT session permutations (165 assignments, LOGO pooled bal_acc):")
    obs_t, p_t, all_t = exact_perm(
        X_t, feats, sessions, make_logistic,
        lambda yp: logo_pooled_bal(X_t, yp, groups, make_logistic))
    print(f"  timing-only:  obs={obs_t:.3f}  exact p={p_t:.3f}")
    obs_b, p_b, _ = exact_perm(
        X_b, feats, sessions, make_logistic,
        lambda yp: logo_pooled_bal(X_b, yp, groups, make_logistic))
    print(f"  BPM-only:     obs={obs_b:.3f}  exact p={p_b:.3f}")
    obs_d, p_d, _ = exact_perm(
        X_tb, feats, sessions, make_logistic,
        lambda yp: (logo_pooled_bal(X_tb, yp, groups, make_logistic)
                    - logo_pooled_bal(X_b, yp, groups, make_logistic)))
    print(f"  complementarity (timing+BPM minus BPM): obs delta={obs_d:+.3f}  "
          f"exact p={p_d:.3f}")
    rows += [
        dict(sample="complete-case", model="timing perm", stat=obs_t, exact_p=p_t),
        dict(sample="complete-case", model="BPM perm", stat=obs_b, exact_p=p_b),
        dict(sample="complete-case", model="delta perm", stat=obs_d, exact_p=p_d),
    ]

    # characterize the null partitions that beat/match the true labeling
    ranked = sorted(all_t, key=lambda t: -t[1])
    true_samba = frozenset(feats.groupby("session").genre.first()
                           .pipe(lambda s: s[s == "samba"]).index)
    print("\n  top null partitions (sessions labeled 'samba'):")
    for ss, s in ranked[:5]:
        mixes = "TRUE LABELING" if ss == true_samba else (
            "corpus-mixing" if any(x.startswith("brid") for x in ss) and
            any(not x.startswith("brid") for x in ss) else "within-corpus")
        print(f"    bal_acc={s:.3f}  {sorted(ss)}  [{mixes}]")

    # commensurable session control: pairwise 2-way session discrimination
    print("\nPAIRWISE session discrimination within corpus "
          "(sessions with >=4 tracks, leave-one-track-out, timing contrasts):")
    for gname in ["candombe", "samba"]:
        sub = feats[feats.genre == gname]
        big = [s for s, n in sub.session.value_counts().items() if n >= 4]
        accs = []
        for s1, s2 in combinations(big, 2):
            pair = sub[sub.session.isin([s1, s2])]
            Xp = pair[ccols].to_numpy()
            yp = (pair.session == s1).to_numpy().astype(int)
            preds = []
            for i in range(len(pair)):
                tr = np.arange(len(pair)) != i
                m = make_logistic()
                m.fit(Xp[tr], yp[tr])
                preds.append(m.predict(Xp[[i]])[0])
            accs.append(balanced_accuracy_score(yp, preds))
        print(f"  {gname:9s}: mean pairwise bal_acc={np.mean(accs):.3f} "
              f"over {len(accs)} pairs (range {min(accs):.2f}-{max(accs):.2f})")
        rows.append(dict(sample="complete-case",
                         model=f"session-pairwise {gname}",
                         lopo_bal=float(np.mean(accs))))

    # sensitivity runs
    print("\nSENSITIVITY:")
    run_suite(full, "full-data (imputed)")
    no_high = [c for c in ccols if "high" not in c]
    bal_nh, auc_nh, _ = lopo_eval(cc[no_high].to_numpy(), y, groups,
                                  feats.groupby("session").genre.first().to_dict(),
                                  make_logistic)
    print(f"  no-high-register (complete-case): bal_acc={bal_nh:.3f} AUC={auc_nh:.3f}")
    rows.append(dict(sample="complete-case", model="timing no-high",
                     lopo_bal=bal_nh, lopo_auc=auc_nh))
    miss = (full[[f"f_{r}_p{p}" for r in REGS for p in [2, 3, 4]]] == 0.0
            ).astype(float).to_numpy()
    y_full = (full.genre == "candombe").to_numpy().astype(int)
    g_full = full.session.to_numpy()
    gog_full = full.groupby("session").genre.first().to_dict()
    bal_m, auc_m, _ = lopo_eval(miss, y_full, g_full, gog_full, make_logistic)
    print(f"  missingness-indicators-only (full data): bal_acc={bal_m:.3f} "
          f"AUC={auc_m:.3f}")
    rows.append(dict(sample="full-data", model="missingness-only",
                     lopo_bal=bal_m, lopo_auc=auc_m))

    # identifiable importances: drop-one-register ablation + logistic coefs
    print("\nIMPORTANCES (complete-case, identifiable):")
    base = logo_pooled_bal(X_t, y, groups, make_logistic)
    abl = {}
    for reg in REGS:
        keep = [c for c in ccols if reg not in c]
        abl[reg] = base - logo_pooled_bal(cc[keep].to_numpy(), y, groups,
                                          make_logistic)
        print(f"  drop-{reg}: delta bal_acc = {abl[reg]:+.3f}")
    m = make_logistic()
    m.fit(X_t, y)
    coefs = dict(zip(ccols, m[-1].coef_[0]))
    print("  standardized logistic coefficients (full fit):")
    for c, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {c:14s} {v:+.2f}")
    for reg in REGS:
        rows.append(dict(sample="complete-case", model=f"ablation drop-{reg}",
                         stat=abl[reg]))

    pd.DataFrame(rows).to_csv(res / "classifier_summary.csv", index=False)
    print(f"\n-> {res / 'classifier_summary.csv'}")

    # ---------------- figures ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    names = ["timing", "BPM", "timing+BPM"]
    vals = [lopo[n]["bal"] for n in names]
    aucs = [lopo[n]["auc"] for n in names]
    x = np.arange(3)
    axes[0].bar(x - 0.18, vals, width=0.36, label="pooled bal_acc", color="#14675A")
    axes[0].bar(x + 0.18, aucs, width=0.36, label="mean per-fold AUC", color="#A5641C")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Leave-one-pair-out CV (complete case)", fontsize=10)
    axes[1].bar(REGS, [abl[r] for r in REGS], color="#14675A", width=0.5)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Δ bal_acc when register removed")
    axes[1].set_title("Register ablation (identifiable importance)", fontsize=10)
    fig.suptitle(
        f"Timing-alone corpus discriminability: exact perm p={p_t:.3f}; "
        f"complementarity Δ={obs_d:+.3f} (p={p_d:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(res / "classifier_accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = sorted(coefs, key=lambda c: -abs(coefs[c]))
    ax.barh([c.replace("c_", "") for c in order][::-1],
            [coefs[c] for c in order][::-1], color="#14675A")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("standardized logistic coefficient (+ = candombe)")
    ax.set_title("Contrast coefficients (6 identifiable features)", fontsize=10)
    fig.tight_layout()
    fig.savefig(res / "classifier_importances.png", dpi=150)
    print("figures -> classifier_accuracy.png, classifier_importances.png")


if __name__ == "__main__":
    main()
