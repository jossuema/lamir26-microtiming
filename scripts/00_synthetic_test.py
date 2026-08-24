"""Synthetic validation of grid + deviation assignment (no audio involved).

Places onsets at known deviations on a constant-tempo grid and checks that
assign_deviations recovers them exactly; also exercises the edge cases
(unassignable onsets, duplicate claims on one grid point, IBI outliers).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from microtiming.attack import attack_starts
from microtiming.deviations import assign_deviations
from microtiming.grid import subdivide_beats

rng = np.random.default_rng(20260820)
failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


# --- grid construction -----------------------------------------------------
beats = np.arange(0.0, 10.0 + 1e-9, 0.5)  # 120 BPM, 21 beats, 20 intervals
grid = subdivide_beats(beats, n_subdivisions=4)
check("grid size = 20 intervals x 4 positions", len(grid.times) == 80, f"(got {len(grid.times)})")
check("grid positions cycle 1..4", list(grid.position[:4]) == [1, 2, 3, 4])
check("grid point spacing = IBI/4", np.allclose(np.diff(grid.times[:4]), 0.125))

# IBI outlier rejection: insert a 3-second gap
beats_gap = np.concatenate([beats[:10], beats[10:] + 3.0])
grid_gap = subdivide_beats(beats_gap, n_subdivisions=4)
check(
    "IBI outlier interval dropped",
    len(grid_gap.times) == 76,
    f"(got {len(grid_gap.times)}, expected 19 intervals x 4)",
)

# --- exact recovery of known deviations ------------------------------------
# put onsets at every grid point with position-dependent shifts, in % of beat:
# pos1: 0%, pos2: +1.5%, pos3: -3%, pos4: -2%  (samba-like anticipation)
true_dev = {1: 0.0, 2: 0.015, 3: -0.03, 4: -0.02}
onset_times = grid.times + np.array([true_dev[p] for p in grid.position]) * grid.beat_duration
strengths = np.ones_like(onset_times)
df = assign_deviations(onset_times, strengths, grid)
check("all synthetic onsets assigned", len(df) == len(grid.times), f"(got {len(df)})")
rec = df.groupby("position")["deviation"].mean()
ok = all(np.isclose(rec[p], true_dev[p], atol=1e-9) for p in [1, 2, 3, 4])
check("deviations recovered exactly", ok, f"(recovered {dict(rec.round(4))})")

# --- unassignable onset dropped --------------------------------------------
lonely = np.array([0.0625 + 0.5 * 7])  # exactly halfway between two grid points
df2 = assign_deviations(lonely, np.ones(1), grid, max_abs_deviation=0.10)
check("midpoint onset dropped (|dev|=12.5% > 10%)", len(df2) == 0)

# --- strongest-wins dedup on one grid point --------------------------------
double = np.array([2.000, 2.010])  # both nearest to the same grid point (t=2.0)
df3 = assign_deviations(double, np.array([0.4, 0.9]), grid)
got = df3[np.isclose(df3.onset_time, 2.010)]
check("strongest onset wins duplicated grid point", len(df3[df3.beat_index == 4]) == 1 and len(got) == 1)

# --- earliest-wins dedup (used for the 'full' ensemble band) ----------------
df3e = assign_deviations(double, np.array([0.4, 0.9]), grid, keep="earliest")
got_e = df3e[np.isclose(df3e.onset_time, 2.000)]
check("earliest onset wins with keep='earliest'", len(df3e[df3e.beat_index == 4]) == 1 and len(got_e) == 1)

# --- noise robustness: jittered onsets, means still recovered --------------
jitter = rng.normal(0.0, 0.004, size=len(onset_times))  # 4 ms SD at 0.5 s beat
df4 = assign_deviations(onset_times + jitter, strengths, grid)
rec4 = df4.groupby("position")["deviation"].mean()
ok4 = all(abs(rec4[p] - true_dev[p]) < 0.005 for p in [1, 2, 3, 4])
check("means recovered under 4 ms jitter (tol 0.5% of beat)", ok4)

# --- attack-start estimator: sharp click vs slow low drum -------------------
# The calibration anchor must locate the attack within ~2 ms on BOTH register
# extremes, or cross-instrument latency comparisons inherit estimator bias.
SR = 22050
T0 = 0.500  # true attack time


def synth(freq: float, decay: float, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    active = t >= T0
    y[active] = np.sin(2 * np.pi * freq * (t[active] - T0)) * np.exp(
        -(t[active] - T0) / decay
    )
    return y + rng.normal(0, 1e-4, len(t))


for label, freq, decay in [("3 kHz click", 3000.0, 0.005), ("60 Hz drum", 60.0, 0.150)]:
    y = synth(freq, decay)
    t_att, found = attack_starts(y, SR, np.array([T0 + 0.010]))
    ok = bool(found[0]) and abs(t_att[0] - T0) <= 0.002
    err_ms = (t_att[0] - T0) * 1000 if found[0] else float("nan")
    check(f"attack estimator on {label} within 2 ms", ok, f"(err={err_ms:.2f} ms)")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all synthetic checks passed")
