# Microtiming as a Multi-Level Fingerprint — Code & Derived Data

Code and derived data accompanying the LAMIR 2026 paper
*"Is Microtiming a Genre Fingerprint? Timing Discriminability in Candombe
and Samba Drumming"*.

**Manuel Josué Malla Campoverde** &middot; **Eduardo Alejandro Tusa Jumbo**
Universidad Técnica de Machala, Ecuador

Presented at the 2nd Latin American Music Information Retrieval Workshop
(LAMIR 2026), Belo Horizonte, Brazil.

## Key results

- Timing shape alone separates the two corpora: balanced accuracy **0.823**
  under leave-one-pair-out CV, exact session-permutation **p = 0.024**.
- Two contrasts — the mid-register *cliff* and the high-register *tail* —
  survive a tempo-controlled permutation ANCOVA under **every**
  cluster-respecting test statistic (p = 0.006, the permutation minimum).
- The same six features identify individual candombe ensembles almost
  perfectly (mean pairwise balanced accuracy **0.964**), so microtiming
  acts as a fingerprint at more than one level of musical identity.

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

## Citation

```bibtex
@inproceedings{malla2026microtiming,
  title     = {Is Microtiming a Genre Fingerprint? Timing Discriminability
               in Candombe and Samba Drumming},
  author    = {Malla Campoverde, Manuel Josu{\'e} and
               Tusa Jumbo, Eduardo Alejandro},
  booktitle = {Proc. of the 2nd Latin American Music Information Retrieval
               Workshop (LAMIR)},
  address   = {Belo Horizonte, Brazil},
  year      = {2026}
}
```
