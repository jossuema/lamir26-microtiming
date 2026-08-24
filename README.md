# Microtiming as a Multi-Level Fingerprint — Code & Derived Data

Anonymized repository accompanying the LAMIR 2026 submission
*"Is Microtiming a Genre Fingerprint? Timing Discriminability in Candombe
and Samba Drumming"*.

## What is here

- `src/microtiming/` — the measurement pipeline: subdivision grid over
  annotated beats (`grid.py`), band-limited onset detection with sub-frame
  interpolation and a mel-power leakage gate (`onsets.py`), deviation
  assignment (`deviations.py`), causal-RMS attack-start anchor
  (`attack.py`), corpus drivers (`pipeline.py`), classifier features
  (`features.py`).
- `scripts/` — the full experimental chain, in order:
  - `00_synthetic_test.py` — synthetic validation (exact recovery of known
    deviations; attack-estimator accuracy on 3 kHz click and 60 Hz drum).
  - `01_extract_deviations.py` — corpora → tidy per-onset deviation tables.
  - `02_sanity_check.py` — profiles vs. published findings (Naveda et al.).
  - `03_calibrate_latency.py` — per-band detector latency vs. acoustic
    attack anchor on BRID solo tracks; band-vs-instrument transfer test.
  - `04_pseudo_mixtures.py` — register-attribution confusion and latency
    transfer measured on pseudo-mixtures with known ground truth.
  - `05_stats.py` — contrast statistics: exact session-label permutation,
    tempo-controlled permutation ANCOVA, Holm correction, variance
    components, tempo/units checks.
  - `06_classifier.py` — discriminability experiment: leave-one-pair-out
    CV, exact 165-assignment permutation inference, baselines and
    sensitivity runs, register ablations.
  - `07_paper_figures.py` — paper figures (vector PDF).
- `results/` — derived data released with the paper: per-onset deviation
  tables (`deviations_*.parquet`), statistics (`stats_summary.csv`),
  classifier results (`classifier_summary.csv`), latency calibration
  (`latency_calibration.json`), attribution measurements
  (`attribution_matrix.csv`, `pseudo_mixture_attribution.csv`).
- `NOTAS_METODO_ES.md` — extended method-decision log (in Spanish),
  documenting every design choice and the measurement caveats behind the
  paper's Methods section.

## Audio data (not included)

The audio corpora are NOT redistributed here. Both are public and download
automatically via [mirdata](https://mirdata.readthedocs.io/):

```python
import mirdata
for name in ["candombe", "brid"]:
    mirdata.initialize(name, data_home=f"./data/{name}").download(cleanup=True)
```

## Setup and reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install mirdata librosa numpy pandas scikit-learn matplotlib soundfile pyarrow scipy
# macOS python.org builds need certifi for downloads:
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")

.venv/bin/python scripts/00_synthetic_test.py   # validation, no audio needed
.venv/bin/python scripts/01_extract_deviations.py
.venv/bin/python scripts/02_sanity_check.py
.venv/bin/python scripts/05_stats.py
.venv/bin/python scripts/06_classifier.py
```

Scripts `03` and `04` (calibration) additionally require the BRID solo
tracks, included in the mirdata download above.

## License

Code under the MIT License (see `LICENSE`). The derived deviation tables
are released under CC BY 4.0. Audio remains under the original corpora's
terms; see the Candombe dataset and BRID publications.

---
*This repository is anonymized for double-blind review.*
