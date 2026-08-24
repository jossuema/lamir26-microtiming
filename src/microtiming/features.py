"""Track-level timing features for the discriminability experiment.

One row per track: for each register (low/mid/high/full), the per-track mean
deviation at positions 2-4, centered over positions 2-4 (2 df per register;
constant per-band detector latency cancels — see README day-2/day-5
decisions). Position 1 is annotation-anchored and NEVER enters features.

Register mapping pairs like registers across corpora:
  candombe: piano->low, repique->mid, chico->high
  samba:    surdo->low, caixa_mid->mid, tamborim->high
"""

import numpy as np
import pandas as pd

REGISTER_MAP = {
    "piano": "low", "repique": "mid", "chico": "high",
    "surdo": "low", "caixa_mid": "mid", "tamborim": "high",
    "full": "full",
}
POSITIONS = [2, 3, 4]


def build_features(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-track feature matrix from one or more deviation tables.

    Missing register x position cells (a track with too few onsets in a band)
    are imputed with 0 = the centered-profile null; the count is reported by
    the caller. Columns: f_{register}_p{position}, plus genre, session,
    mean_bpm, dataset metadata.
    """
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.position.isin(POSITIONS)].copy()
    df["register"] = df.band.map(REGISTER_MAP)

    g = (
        df.groupby(["track_id", "register", "position"])
        .deviation.mean()
        .unstack()
    )
    g = g.sub(g.mean(axis=1), axis=0) * 100  # center over pos 2-4, % of beat
    wide = g.unstack("register")
    wide.columns = [f"f_{reg}_p{pos}" for pos, reg in wide.columns]

    meta = df.groupby("track_id").agg(
        genre=("genre", "first"), subgenre=("subgenre", "first"),
        session=("session", "first"), mean_bpm=("mean_bpm", "first"),
        dataset=("dataset", "first"),
    )
    out = wide.merge(meta, left_index=True, right_index=True)
    feat_cols = [c for c in out.columns if c.startswith("f_")]
    out["n_missing"] = out[feat_cols].isna().sum(axis=1)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out.reset_index()
