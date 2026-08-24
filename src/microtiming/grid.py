"""Isochronous subdivision grid built from annotated beats.

The grid is the *reference* against which deviations are measured, following
Jure & Rocamora (2016) and Fuentes et al. (2019): within each inter-beat
interval, K equally spaced subdivision positions are laid down (K=4 for the
16th-note level in 4/4 candombe and 2/4 samba at the tatum level).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SubdivisionGrid:
    """Subdivision grid over one track.

    times[i]      : expected (isochronous) time of grid point i
    beat_index[i] : index of the beat interval the point belongs to (0-based)
    position[i]   : subdivision position within the beat, 1..K (1 = on the beat)
    beat_duration[i] : duration of the enclosing inter-beat interval (seconds)
    """

    times: np.ndarray
    beat_index: np.ndarray
    position: np.ndarray
    beat_duration: np.ndarray
    n_subdivisions: int


def subdivide_beats(
    beat_times: np.ndarray,
    n_subdivisions: int = 4,
    min_ibi: float = 0.25,
    max_ibi: float = 1.0,
) -> SubdivisionGrid:
    """Lay an isochronous K-subdivision grid over annotated beats.

    Inter-beat intervals outside [min_ibi, max_ibi] seconds are discarded:
    they indicate annotation gaps, pickups, or segment boundaries rather than
    real musical beats (0.25-1.0 s spans 60-240 BPM).
    """
    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.ndim != 1 or len(beat_times) < 2:
        raise ValueError("beat_times must be a 1-D array with at least 2 beats")

    times, beat_idx, position, durations = [], [], [], []
    ibis = np.diff(beat_times)
    for i, ibi in enumerate(ibis):
        if not (min_ibi <= ibi <= max_ibi):
            continue
        for k in range(n_subdivisions):
            times.append(beat_times[i] + ibi * k / n_subdivisions)
            beat_idx.append(i)
            position.append(k + 1)
            durations.append(ibi)

    return SubdivisionGrid(
        times=np.asarray(times),
        beat_index=np.asarray(beat_idx, dtype=int),
        position=np.asarray(position, dtype=int),
        beat_duration=np.asarray(durations),
        n_subdivisions=n_subdivisions,
    )
