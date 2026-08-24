"""Day-2 calibration on BRID solo tracks — v2 after adversarial review.

v1 flaws fixed here (see README decisions):
  - anchor: the full-band flux peak was instrument-dependent (on a solo it IS
    the instrument), so band-vs-full "differentials" were near-degenerate.
    v2 anchors every band onset to the ACOUSTIC ATTACK START on the waveform
    (causal-RMS estimator, validated to <1 ms on synthetic 3 kHz click and
    60 Hz drum): absolute latency B_b(X) = t_band_onset - t_attack.
  - no cross-detector match window (v1's +/-20 ms censored slow-attack tails):
    the attack is searched around each band onset directly; only a [0, 60 ms]
    plausibility bound applies, with the discard fraction reported.
  - gate: run WITH the fixed power-ratio gate (gate_db=20). On a solo the own
    band always dominates; another band passes only if genuinely excited
    (e.g. caixa in both mid and high), so surviving offsets are physical.
  - estimator sensitivity: rise_frac in {0.05, 0.10, 0.20}; if per-instrument
    medians move > 2 ms across thresholds, the numbers bound rather than
    correct. Skip rates reported (ringing/rolls are excluded by hysteresis).

Transfer criterion (candombe has no solo stems): per band, across instruments
that genuinely excite it (>= 5 tracks with >= 15 valid onsets), the max
pairwise difference of per-instrument medians must be <= 2 ms (0.4% of a
500 ms beat) for latency to count as band-driven. Reported explicitly.

Outputs: results/latency_calibration.json, results/attribution_matrix.csv.
"""

import json
import sys
from collections import defaultdict
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
MIN_VALID = 15
PLAUSIBLE_MS = (0.0, 60.0)
RISE_FRACS = (0.05, 0.10, 0.20)
GATE_DB = 20.0


def main() -> None:
    ds = mirdata.initialize("brid", data_home=str(ROOT / "data/brid"))
    solos = []
    for tid in ds.track_ids:
        t = ds.track(tid)
        if t.beats_path is None:
            solos.append((tid, Path(t.audio_path).parent.name, t.audio_path))
    print(f"{len(solos)} solo tracks | gate_db={GATE_DB} | anchor=causal-RMS attack start")

    bands = {**SAMBA_BANDS, **FULL_BAND}
    role_bands = [b for b in bands if b != "full"]

    rows = []
    attribution = defaultdict(lambda: defaultdict(int))
    skip_stats = defaultdict(lambda: [0, 0])       # instrument -> [skipped, events]
    discard_stats = defaultdict(lambda: [0, 0])    # instrument -> [implausible, found]

    for k, (tid, instrument, path) in enumerate(solos):
        y, sr = librosa.load(path, sr=SR, mono=True)
        env = causal_rms(y, sr)
        det = detect_band_onsets(y, sr, bands, gate_db=GATE_DB)

        for b in role_bands:
            t_band = det[b][0]
            attribution[instrument][b] += len(t_band)
            if len(t_band) < MIN_VALID:
                continue
            med = {}
            for rf in RISE_FRACS:
                t_att, found = attack_starts(y, sr, t_band, rise_frac=rf, env=env)
                skip_stats[instrument][0] += int((~found).sum())
                skip_stats[instrument][1] += len(found)
                off_ms = (t_band[found] - t_att[found]) * 1000
                ok = (off_ms >= PLAUSIBLE_MS[0]) & (off_ms <= PLAUSIBLE_MS[1])
                discard_stats[instrument][0] += int((~ok).sum())
                discard_stats[instrument][1] += len(off_ms)
                if ok.sum() >= MIN_VALID:
                    med[rf] = float(np.median(off_ms[ok]))
            if 0.10 in med:
                rows.append(
                    {
                        "instrument": instrument,
                        "band": b,
                        "track_id": tid,
                        "latency_ms": med[0.10],
                        "sens_low": med.get(0.05, np.nan),
                        "sens_high": med.get(0.20, np.nan),
                    }
                )
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(solos)}", flush=True)

    lat = pd.DataFrame(rows)

    print("\nABSOLUTE per-band latency vs attack start (ms), per instrument")
    summary = (
        lat.groupby(["instrument", "band"])["latency_ms"]
        .agg(["median", "std", "count"])
        .round(2)
    )
    print(summary)

    print("\nEstimator sensitivity (median shift of latency_ms, rise 0.05 vs 0.20):")
    sens = lat.dropna(subset=["sens_low", "sens_high"]).groupby("instrument").apply(
        lambda g: (g.sens_high - g.sens_low).median(), include_groups=False
    )
    print(sens.round(2))

    print("\nSkip rate (hysteresis: ringing/rolls) and implausible-offset rate:")
    for instr in sorted(skip_stats):
        s, n = skip_stats[instr]
        d, m = discard_stats[instr]
        print(
            f"  {instr:10s} skipped {s}/{n} ({s / max(n, 1):.0%})"
            f", implausible {d}/{m} ({d / max(m, 1):.0%})"
        )

    att = pd.DataFrame(attribution).T.fillna(0).astype(int)
    att_share = att.div(att.sum(axis=1), axis=0).round(3)
    att_share.index.name = "instrument"
    print("\nATTRIBUTION diagnostic on solos (share of gated onsets per band)")
    print("(diagnostic only — real mixture confusion comes from pseudo-mixtures)")
    print(att_share)
    att_share.to_csv(ROOT / "results/attribution_matrix.csv")

    # transfer criterion: band-driven vs instrument-driven
    print("\nTRANSFER criterion per band (max pairwise diff of instrument medians, ms):")
    transfer = {}
    for b in role_bands:
        sub = lat[lat.band == b].groupby("instrument")["latency_ms"].agg(["median", "count"])
        sub = sub[sub["count"] >= 5]
        if len(sub) >= 2:
            spread = float(sub["median"].max() - sub["median"].min())
            verdict = "band-driven (<=2ms)" if spread <= 2.0 else "INSTRUMENT-driven (>2ms)"
            print(f"  {b:10s} {spread:5.2f} ms across {len(sub)} instruments -> {verdict}")
            transfer[b] = {"spread_ms": spread, "n_instruments": int(len(sub)),
                           "band_driven": spread <= 2.0}
        else:
            print(f"  {b:10s} insufficient instruments with >=5 tracks")
            transfer[b] = {"spread_ms": None, "n_instruments": int(len(sub)),
                           "band_driven": False}

    role_of = {"surdo": "Surdo", "caixa_mid": "Caixa", "tamborim": "Tamborim"}
    calib = {}
    for band, instr in role_of.items():
        own = lat[(lat.band == band) & (lat.instrument == instr)]
        calib[band] = {
            "latency_ms": float(own.latency_ms.median()) if len(own) else None,
            "n_tracks": int(len(own)),
        }
    out = {
        "anchor": "causal-RMS attack start (validated <1ms on synthetic pulses)",
        "gate_db": GATE_DB,
        "scope": (
            "corrects RAW cross-role comparisons within BRID only; centered "
            "profiles are invariant to constant per-band offsets, so "
            "cross-genre claims stay centered + like-register and need no "
            "correction; transfer to candombe requires band_driven=true AND "
            "pseudo-mixture validation (script 04)"
        ),
        "role_latency_ms": calib,
        "transfer": transfer,
        "per_instrument_band_ms": {
            f"{i}|{b}": float(g.latency_ms.median())
            for (i, b), g in lat.groupby(["instrument", "band"])
        },
    }
    with open(ROOT / "results/latency_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\ncalibration -> results/latency_calibration.json")


if __name__ == "__main__":
    main()
