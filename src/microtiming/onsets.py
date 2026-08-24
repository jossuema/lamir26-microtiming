"""Band-wise onset detection for percussion ensembles.

Instrument-role attribution in mixtures follows the register-based strategy:
each candombe drum (and each samba timekeeper) occupies a distinct spectral
register, so onsets detected on a band-limited onset-strength envelope can be
attributed to a role. Band edges are configuration, not fact — they get tuned
against solo/stem material and reported in the paper.

Timing precision: microtiming effects are 1-4% of a ~0.5 s beat (5-20 ms), so
onset times must be finer than that. We use hop_length=64 at sr=22050
(2.9 ms/frame) and refine each detected peak with parabolic interpolation of
the onset-strength envelope, giving sub-frame (sub-millisecond) onset times.

Known limitation (calibrate before cross-ROLE claims): detection latency is
band-dependent — low drums have slow attacks, high drums sharp ones — which
offsets roles against each other by the order of the effect size. The
`latency_s` per-band offsets exist to hold calibration values measured on
solo/stem tracks; until calibrated, compare like registers across genres
(surdo vs piano, tamborim vs chico), which share detector bias.
"""

import numpy as np
import librosa

# Register bands (Hz). Candombe roles after Jure & Rocamora's register
# description (piano = low drum, repique = mid, chico = high); "full" is the
# unattributed mixture-level envelope.
CANDOMBE_BANDS = {
    "piano": (20.0, 200.0),
    "repique": (200.0, 1200.0),
    "chico": (1200.0, 8000.0),
}
SAMBA_BANDS = {
    "surdo": (20.0, 150.0),
    "caixa_mid": (150.0, 1500.0),
    "tamborim": (1500.0, 9000.0),
}
FULL_BAND = {"full": (20.0, 11025.0)}


def band_onset_strength(
    y: np.ndarray,
    sr: float,
    bands: dict,
    hop_length: int = 64,
    n_fft: int = 1024,
    n_mels: int = 128,
) -> tuple[dict, dict, float]:
    """Compute onset-strength and mean power envelopes per frequency band.

    Returns ({band: flux_envelope}, {band: power_db_envelope}, fps). A single
    mel spectrogram is computed once and sliced per band, so bands stay
    mutually consistent. Rows are selected by the TRUE mel filter center
    frequencies (mel_frequencies(n_mels + 2)[1:-1]), not the grid endpoints.

    power_db is the MEAN power per mel row in the band (dB), so band width
    does not inflate it — this is the common scale the leakage gate compares
    across bands. The flux envelope (dB-domain spectral flux) is nearly
    level-invariant and must NOT be compared across bands for dominance.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        n_mels=n_mels, fmin=20.0, fmax=sr / 2,
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=None)
    mel_centers = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=20.0, fmax=sr / 2)[1:-1]

    envelopes, power_db = {}, {}
    for name, (fmin, fmax) in bands.items():
        rows = np.where((mel_centers >= fmin) & (mel_centers <= fmax))[0]
        if len(rows) == 0:
            raise ValueError(f"band {name} ({fmin}-{fmax} Hz) selects no mel rows")
        envelopes[name] = librosa.onset.onset_strength(
            S=S_db[rows], sr=sr, hop_length=hop_length
        )
        power_db[name] = 10.0 * np.log10(S[rows].mean(axis=0) + 1e-12)
    return envelopes, power_db, sr / hop_length


def _parabolic_refine(env: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-frame peak positions via parabolic interpolation of the envelope."""
    p = peaks.astype(float)
    inner = (peaks > 0) & (peaks < len(env) - 1)
    a = env[peaks[inner] - 1]
    b = env[peaks[inner]]
    c = env[peaks[inner] + 1]
    den = a - 2 * b + c
    shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    p[inner] += np.clip(shift, -0.5, 0.5)
    return p


def detect_band_onsets(
    y: np.ndarray,
    sr: float,
    bands: dict,
    hop_length: int = 64,
    n_fft: int = 1024,
    delta: float = 0.07,
    pre_max_s: float = 0.015,
    avg_s: float = 0.09,
    wait_s: float = 0.03,
    gate_db: float | None = 20.0,
    latency_s: dict | None = None,
) -> dict:
    """Detect onsets per band. Returns {band_name: (onset_times, onset_strengths)}.

    Peak-picking windows are specified in SECONDS and converted to frames, so
    behavior does not silently change with hop_length. For PEAK PICKING each
    flux envelope is normalized by its own 95th percentile (robust to one
    spurious transient) — that per-band scale is fine for picking WITHIN a
    band but meaningless across bands, so the leakage gate never uses it.

    Cross-band leakage gate (power-ratio): percussive attacks are broadband,
    so a loud drum raises even a far register's envelope — and dB-domain
    flux is nearly level-invariant, so flux comparisons cannot detect this.
    The gate instead compares MEAN PER-ROW MEL POWER (dB, common scale): a
    peak in band b is kept only if b's power within +/-2 frames of the peak
    is at least (max over other instrument bands) - gate_db. True leakage
    sits tens of dB down, so gate_db=20 rejects it while genuinely sounding
    instruments (within 20 dB of the loudest band) pass. The "full" band is
    exempt: it is the unattributed ensemble envelope by design. gate_db=None
    disables the gate.

    `latency_s` maps band name -> detection-latency offset (seconds) to be
    SUBTRACTED from onset times once calibrated on solo material; defaults 0.
    """
    envelopes, power_db, fps = band_onset_strength(
        y, sr, bands, hop_length=hop_length, n_fft=n_fft
    )
    latency_s = latency_s or {}

    norm = {}
    for name, env in envelopes.items():
        scale = np.percentile(env[env > 0], 95) if np.any(env > 0) else 1.0
        norm[name] = env / (scale + 1e-9)

    to_frames = lambda s: max(1, int(round(s * fps)))
    out = {}
    for name, env_n in norm.items():
        peaks = librosa.util.peak_pick(
            env_n,
            pre_max=to_frames(pre_max_s), post_max=to_frames(pre_max_s),
            pre_avg=to_frames(avg_s), post_avg=to_frames(avg_s),
            delta=delta, wait=to_frames(wait_s),
        )
        if name != "full" and gate_db is not None and len(peaks):
            other_bands = [o for o in power_db if o not in (name, "full")]
            if other_bands:
                n_frames = len(power_db[name])
                keep = np.ones(len(peaks), dtype=bool)
                for k, p in enumerate(peaks):
                    lo, hi = max(0, p - 2), min(n_frames, p + 3)
                    own = power_db[name][lo:hi].max()
                    other = max(power_db[o][lo:hi].max() for o in other_bands)
                    keep[k] = own >= other - gate_db
                peaks = peaks[keep]
        refined = _parabolic_refine(envelopes[name], peaks)
        times = refined * hop_length / sr - latency_s.get(name, 0.0)
        out[name] = (times, env_n[peaks])
    return out
