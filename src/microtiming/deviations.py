"""Assign onsets to grid positions and compute normalized timing deviations."""

import numpy as np
import pandas as pd

from .grid import SubdivisionGrid


def assign_deviations(
    onset_times: np.ndarray,
    onset_strengths: np.ndarray,
    grid: SubdivisionGrid,
    max_abs_deviation: float = 0.125,
    keep: str = "strongest",
) -> pd.DataFrame:
    """Match each onset to its nearest grid point and compute the deviation.

    deviation = (t_onset - t_grid) / beat_duration, so it is expressed as a
    fraction of the beat (negative = ahead of the grid / anticipation).

    The default max_abs_deviation of 0.125 is the full Voronoi cell at K=4
    subdivisions (spacing 0.25 beats): nothing inside a cell is truncated, so
    the anticipation tail of the distribution — the effect under study, which
    reaches 5-9% of the beat in samba — is never censored. Pass a smaller
    value only for sensitivity analyses, and report it.

    One onset may claim each grid point at most once. keep="strongest" keeps
    the highest-strength onset (per-role bands: accented stroke wins over
    flams/ghosts); keep="earliest" keeps the first in time (use for the
    unattributed "full" band, where strongest-wins would turn the ensemble
    profile into a loudest-instrument profile).
    """
    if keep not in ("strongest", "earliest"):
        raise ValueError(f"keep must be 'strongest' or 'earliest', got {keep!r}")
    onset_times = np.asarray(onset_times, dtype=float)
    onset_strengths = np.asarray(onset_strengths, dtype=float)
    if len(grid.times) == 0 or len(onset_times) == 0:
        return pd.DataFrame(
            columns=["beat_index", "position", "deviation", "onset_time", "onset_strength"]
        )

    idx = np.searchsorted(grid.times, onset_times)
    idx_left = np.clip(idx - 1, 0, len(grid.times) - 1)
    idx_right = np.clip(idx, 0, len(grid.times) - 1)
    d_left = np.abs(onset_times - grid.times[idx_left])
    d_right = np.abs(onset_times - grid.times[idx_right])
    nearest = np.where(d_left <= d_right, idx_left, idx_right)

    deviation = (onset_times - grid.times[nearest]) / grid.beat_duration[nearest]
    within = np.abs(deviation) <= max_abs_deviation

    df = pd.DataFrame(
        {
            "grid_point": nearest[within],
            "beat_index": grid.beat_index[nearest[within]],
            "position": grid.position[nearest[within]],
            "deviation": deviation[within],
            "onset_time": onset_times[within],
            "onset_strength": onset_strengths[within],
        }
    )
    # one onset per grid point: strongest or earliest wins
    sort_col = "onset_strength" if keep == "strongest" else "onset_time"
    ascending = keep == "earliest"
    df = (
        df.sort_values(sort_col, ascending=ascending)
        .drop_duplicates(subset="grid_point", keep="first")
        .sort_values("onset_time")
        .drop(columns="grid_point")
        .reset_index(drop=True)
    )
    return df
