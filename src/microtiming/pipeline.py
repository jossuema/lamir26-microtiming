"""Corpus-level extraction: mirdata tracks -> tidy deviation table."""

import os

import librosa
import numpy as np
import pandas as pd

from .deviations import assign_deviations
from .grid import subdivide_beats
from .onsets import CANDOMBE_BANDS, FULL_BAND, SAMBA_BANDS, detect_band_onsets


def process_track(
    audio_path: str,
    beat_times: np.ndarray,
    bands: dict,
    track_meta: dict,
    sr: float = 22050,
    n_subdivisions: int = 4,
) -> pd.DataFrame:
    """Run the full per-track pipeline: audio -> band onsets -> deviations."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    grid = subdivide_beats(beat_times, n_subdivisions=n_subdivisions)
    band_onsets = detect_band_onsets(y, sr, bands)

    frames = []
    for band, (times, strengths) in band_onsets.items():
        # "full" is the unattributed ensemble envelope: earliest onset per grid
        # point, otherwise strongest-wins would track the loudest instrument.
        df = assign_deviations(
            times, strengths, grid, keep="earliest" if band == "full" else "strongest"
        )
        df["band"] = band
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for key, value in track_meta.items():
        out[key] = value
    # tempo covariate from the SAME intervals the grid retained, so it stays
    # consistent with the deviations (raw IBIs may include annotation gaps)
    out["mean_bpm"] = 60.0 / grid.beat_duration.mean()
    return out


def _run_corpus(jobs: list, corpus_name: str) -> pd.DataFrame:
    """Run per-track jobs, keep going on failure, and report loudly at the end.

    Failures are printed (not warned) so no warnings filter can hide them.
    """
    frames, failed = [], []
    for track_id, fn in jobs:
        try:
            frames.append(fn())
        except Exception as exc:
            failed.append(track_id)
            print(f"[{corpus_name}] FAILED {track_id}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"[{corpus_name}] {len(failed)}/{len(jobs)} tracks failed: {failed}")
    if not frames:
        raise RuntimeError(
            f"[{corpus_name}] no tracks processed ({len(failed)} failed). "
            f"Is the dataset downloaded and extracted? Run dataset.download()."
        )
    return pd.concat(frames, ignore_index=True)


def process_candombe(dataset, limit: int | None = None) -> pd.DataFrame:
    """Extract deviations for the Candombe dataset (35 ensemble recordings)."""
    bands = {**CANDOMBE_BANDS, **FULL_BAND}
    ids = dataset.track_ids[:limit] if limit else dataset.track_ids

    def make_job(track_id):
        def job():
            track = dataset.track(track_id)
            if not os.path.exists(track.audio_path):
                raise FileNotFoundError(f"audio missing: {track.audio_path}")
            beat_times = track.beats.times  # inside try: missing csv -> skip+report
            meta = {
                "dataset": "candombe",
                "genre": "candombe",
                "subgenre": "candombe",
                "track_id": track_id,
                # ensemble/session unit for leakage-safe grouped CV, e.g.
                # "csic.1995_ansina1" (8 groups) — NOT the 3 mega-sessions
                "session": track_id.rsplit("_", 1)[0],
            }
            return process_track(track.audio_path, beat_times, bands, meta)
        return track_id, job

    return _run_corpus([make_job(t) for t in ids], "candombe")


# BRID rhythm-class codes -> (subgenre label, genre family)
_BRID_RHYTHMS = {
    "SA": ("samba", "samba"),
    "SE": ("samba de enredo", "samba"),
    "PA": ("partido alto", "samba"),
    "MA": ("marcha", "marcha"),
    "CA": ("capoeira", "capoeira"),
    "BA": ("baiao", "baiao"),
    "MX": ("maxixe", "maxixe"),
}


def process_brid_mixtures(dataset, limit: int | None = None) -> pd.DataFrame:
    """Extract deviations for BRID's beat-annotated ensemble mixtures.

    Only tracks with beat annotations are processed (the 93 mixtures; solo
    tracks have beats=None in the index). genre is set from the rhythm class:
    BRID's annotated mixtures include a few non-samba rhythms (e.g. marcha)
    that must NOT be pooled into samba profiles — filter genre == "samba"
    downstream and report the counts.
    """
    bands = {**SAMBA_BANDS, **FULL_BAND}
    ids = dataset.track_ids[:limit] if limit else dataset.track_ids

    jobs, skipped = [], 0
    for track_id in ids:
        track = dataset.track(track_id)
        if track.beats_path is None:  # solo track: no beat annotations
            skipped += 1
            continue
        jobs.append((track_id, _make_brid_job(dataset, track_id, bands)))
    print(f"BRID: {len(jobs)} beat-annotated mixtures queued, {skipped} solo tracks skipped")

    df = _run_corpus(jobs, "brid")
    counts = df.groupby("subgenre")["track_id"].nunique()
    print(f"BRID rhythm classes (tracks): {counts.to_dict()}")
    return df


def _make_brid_job(dataset, track_id: str, bands: dict):
    def job():
        track = dataset.track(track_id)
        if not os.path.exists(track.audio_path):
            raise FileNotFoundError(f"audio missing: {track.audio_path}")
        beats = track.beats
        if beats is None or beats.times is None or len(beats.times) < 2:
            raise ValueError("beat annotation empty or too short")
        subgenre, genre = _brid_rhythm_class(track, track_id)
        meta = {
            "dataset": "brid",
            "genre": genre,
            "subgenre": subgenre,
            "track_id": track_id,
            # musician-set token, e.g. "[0042] M3-23-SE" -> "M3": the grouping
            # unit for leakage-safe grouped CV on the BRID side
            "session": "brid_" + track_id.split()[-1].split("-")[0],
        }
        return process_track(track.audio_path, beats.times, bands, meta)
    return job


def _brid_rhythm_class(track, track_id: str) -> tuple[str, str]:
    code = None
    rhythm = getattr(track, "rhythm", None)
    if rhythm:
        # mirdata may expose the full name; normalize to our labels
        r = str(rhythm).strip().lower()
        for sub, fam in _BRID_RHYTHMS.values():
            if r == sub:
                return sub, fam
    # fall back to the id suffix, e.g. "[0042] M3-23-SE" -> SE
    code = track_id.strip().rsplit("-", 1)[-1].strip().upper()
    return _BRID_RHYTHMS.get(code, (code.lower(), code.lower()))
