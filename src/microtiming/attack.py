"""Physical attack-start estimation — the common anchor for latency calibration.

The detector's flux peak lags the physical strike by a band- and
instrument-dependent amount. To measure that latency we need a reference that
does NOT depend on the detector: the acoustic attack start, estimated on a
CAUSAL RMS envelope of the waveform (window over past samples only). A causal
envelope cannot rise before the attack — symmetric smoothing smears energy
backward, and the Hilbert envelope pre-rings at low fundamentals, both of
which bias attack estimates EARLY by ms (measured on synthetic pulses). With
a causal window the 10%-of-peak crossing lands <1 ms after the true start for
both a 3 kHz click and a 60 Hz drum tone; scripts/00_synthetic_test.py
enforces 2 ms on both extremes.

Hysteresis rule per event: floor = 5th percentile of the envelope in the
pre-window; the attack start is the first envelope crossing of
floor + rise_frac*(peak - floor) AFTER the last time the envelope was below
floor + floor_frac*(peak - floor). Events whose envelope never drops below
the low threshold before the peak (continuous ringing, rolls) are skipped and
counted — report the skip rate; a biased sample is worse than a small one.
"""

import numpy as np


def causal_rms(y: np.ndarray, sr: float, window_s: float = 0.005) -> np.ndarray:
    """RMS over the PAST window_s seconds at each sample (causal, no lookahead)."""
    w = max(1, int(round(window_s * sr)))
    kernel = np.ones(w) / w
    mean_sq = np.convolve(y**2, kernel, mode="full")[: len(y)]
    return np.sqrt(mean_sq)


def attack_starts(
    y: np.ndarray,
    sr: float,
    t_events: np.ndarray,
    pre_s: float = 0.08,
    post_s: float = 0.05,
    rise_frac: float = 0.10,
    floor_frac: float = 0.05,
    window_s: float = 0.005,
    env: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attack-start time for each approximate event time.

    Returns (t_attack, found): t_attack aligned with t_events (NaN where the
    hysteresis condition failed), found = boolean mask. Pass a precomputed
    `env` (from causal_rms) to amortize cost across calls.
    """
    if env is None:
        env = causal_rms(y, sr, window_s=window_s)
    n = len(env)
    t_attack = np.full(len(t_events), np.nan)
    for j, t in enumerate(np.asarray(t_events, dtype=float)):
        i = int(round(t * sr))
        lo = max(0, i - int(pre_s * sr))
        hi = min(n, i + int(post_s * sr))
        if hi - lo < int(0.02 * sr):
            continue
        seg = env[lo:hi]
        peak_rel = int(np.argmax(seg))
        peak = seg[peak_rel]
        floor = np.percentile(seg[: max(1, peak_rel)], 5) if peak_rel > 0 else 0.0
        if peak <= floor:
            continue
        th_low = floor + floor_frac * (peak - floor)
        th_rise = floor + rise_frac * (peak - floor)
        pre = seg[: peak_rel + 1]
        below = np.where(pre < th_low)[0]
        if len(below) == 0:
            continue  # never quiet before the peak: ringing/roll — skip
        start = below[-1]
        risen = np.where(pre[start:] >= th_rise)[0]
        if len(risen) == 0:
            continue
        t_attack[j] = (lo + start + risen[0]) / sr
    found = ~np.isnan(t_attack)
    return t_attack, found
