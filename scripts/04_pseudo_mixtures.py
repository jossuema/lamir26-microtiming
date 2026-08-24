"""Attribution + latency-transfer validation on pseudo-mixtures — v2.

Ground truth per instrument = ATTACK STARTS on the isolated solo (causal-RMS
estimator, same anchor as script 03), located around the solo's own full-band
onsets. Mixtures = RMS-balanced sums of surdo + caixa + tamborim solos of the
same rhythm class (BRID solos were recorded to a standardized click).

Measures, per band and gate strength (gate_db sweep):
  precision_own — band onsets matching the band's OWN instrument's truth
  leak_other    — band onsets matching a DIFFERENT instrument's truth
                  (ambiguous coincidences count in both)
  recall_own    — truth onsets recovered by the band
  mix_latency   — median (t_band_onset - matched own truth attack) in the
                  MIXTURE; compare against script 03's solo-measured latency:
                  the disagreement is the calibration's transfer uncertainty.

Caveat (report in the paper): solos are not phase-aligned, so coincident hits
— where attribution is hardest — are rarer than in real ensembles.

Outputs: results/pseudo_mixture_attribution.csv + printed summary.
"""

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import mirdata
import numpy as np
import pandas as pd

from microtiming.attack import attack_starts, causal_rms
from microtiming.onsets import FULL_BAND, SAMBA_BANDS, detect_band_onsets

SR = 22050
MATCH_S = 0.020
ROLE_OF = {"surdo": "Surdo", "caixa_mid": "Caixa", "tamborim": "Tamborim"}
N_TRIPLETS_PER_CLASS = 12
GATE_SWEEP = [None, 30.0, 20.0, 10.0]

rng = np.random.default_rng(20260821)


def load_solos(ds):
    by_class = {}
    for tid in ds.track_ids:
        t = ds.track(tid)
        if t.beats_path is not None:
            continue
        instr = Path(t.audio_path).parent.name
        if instr not in ("Surdo", "Caixa", "Tamborim"):
            continue
        rhythm = tid.strip().rsplit("-", 1)[-1]
        if rhythm in ("SA", "SE", "PA"):
            by_class.setdefault(rhythm, {}).setdefault(instr, []).append(t.audio_path)
    return by_class


def truth_onsets(y):
    """Two truth references from an isolated solo.

    - flux:   full-band flux-peak onsets. Use for ATTRIBUTION matching: band
      onsets in the mixture carry the same ~30 ms detector latency as these,
      so it cancels and a +/-20 ms window is meaningful.
    - attack: acoustic attack starts (same anchor as script 03). Use for
      LATENCY: t_band - t_attack with a lag-aware window — matching attacks
      within +/-20 ms would censor the ~30 ms systematic lag itself.
    """
    det = detect_band_onsets(y, SR, dict(FULL_BAND), gate_db=None)
    t_flux = det["full"][0]
    t_att, found = attack_starts(y, SR, t_flux, env=causal_rms(y, SR))
    return {"flux": np.sort(t_flux), "attack": np.sort(t_att[found])}


def nearest_offsets(t_query, t_ref):
    """Signed offset to the nearest reference for each query time."""
    if len(t_query) == 0 or len(t_ref) == 0:
        return np.full(len(t_query), np.inf)
    idx = np.searchsorted(t_ref, t_query)
    idx_l = np.clip(idx - 1, 0, len(t_ref) - 1)
    idx_r = np.clip(idx, 0, len(t_ref) - 1)
    d_l = t_query - t_ref[idx_l]
    d_r = t_query - t_ref[idx_r]
    return np.where(np.abs(d_l) <= np.abs(d_r), d_l, d_r)


def rms_normalize(y):
    return y / (np.sqrt(np.mean(y**2)) + 1e-9)


def main() -> None:
    ds = mirdata.initialize("brid", data_home=str(ROOT / "data/brid"))
    by_class = load_solos(ds)

    rows = []
    for rhythm, pools in sorted(by_class.items()):
        triplets = list(product(pools["Surdo"], pools["Caixa"], pools["Tamborim"]))
        take = min(N_TRIPLETS_PER_CLASS, len(triplets))
        chosen = [triplets[i] for i in rng.choice(len(triplets), take, replace=False)]
        print(f"{rhythm}: {len(triplets)} possible triplets, testing {take}", flush=True)

        for paths in chosen:
            ys = [librosa.load(p, sr=SR, mono=True)[0] for p in paths]
            n = min(map(len, ys))
            ys = [rms_normalize(y[:n]) for y in ys]
            refs = {instr: truth_onsets(y) for instr, y in zip(ROLE_OF.values(), ys)}
            mix = sum(ys)

            for gate in GATE_SWEEP:
                det = detect_band_onsets(mix, SR, {**SAMBA_BANDS, **FULL_BAND}, gate_db=gate)
                for band, own in ROLE_OF.items():
                    t_band = det[band][0]
                    # attribution vs FLUX truth (shared latency cancels)
                    off_own = nearest_offsets(t_band, refs[own]["flux"])
                    own_hit = np.abs(off_own) <= MATCH_S
                    other_hit = sum(
                        int((np.abs(nearest_offsets(t_band, refs[i]["flux"])) <= MATCH_S).sum())
                        for i in refs if i != own
                    )
                    rec = int(
                        (np.abs(nearest_offsets(refs[own]["flux"], t_band)) <= MATCH_S).sum()
                    ) if len(t_band) else 0
                    # latency vs ATTACK truth, lag-aware window [-5, 60] ms
                    off_att = nearest_offsets(t_band, refs[own]["attack"]) * 1000
                    lag_ok = (off_att >= -5.0) & (off_att <= 60.0)
                    lat = (
                        float(np.median(off_att[lag_ok]))
                        if lag_ok.sum() >= 10 else np.nan
                    )
                    rows.append(
                        {
                            "rhythm": rhythm,
                            "gate_db": -1.0 if gate is None else gate,
                            "band": band,
                            "n_band_onsets": len(t_band),
                            "n_ref": len(refs[own]["flux"]),
                            "own_matched": int(own_hit.sum()),
                            "other_matched": other_hit,
                            "ref_recovered": rec,
                            "mix_latency_ms": lat,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/pseudo_mixture_attribution.csv", index=False)

    agg = df.groupby(["gate_db", "band"]).sum(numeric_only=True)
    agg["precision_own"] = agg.own_matched / agg.n_band_onsets.clip(lower=1)
    agg["leak_other"] = agg.other_matched / agg.n_band_onsets.clip(lower=1)
    agg["recall_own"] = agg.ref_recovered / agg.n_ref.clip(lower=1)
    agg["mix_latency_ms"] = df.groupby(["gate_db", "band"])["mix_latency_ms"].median()
    print("\nATTRIBUTION + LATENCY on pseudo-mixtures (gate_db=-1 means gate off)")
    print(
        agg[["n_band_onsets", "precision_own", "leak_other", "recall_own", "mix_latency_ms"]]
        .round(3)
        .to_string()
    )
    print("\n-> results/pseudo_mixture_attribution.csv")


if __name__ == "__main__":
    main()
