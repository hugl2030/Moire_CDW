# Moire CDW

Moiré diffraction analysis with a simple workflow:
**find peaks and vectors → fitting → calibration**. Three-lattice analysis is separate.

## Code

| File | Purpose |
| --- | --- |
| `src/moire_cdw/superlattice.py` | Peak detection and automatic two-lattice vectors |
| `src/moire_cdw/fitting.py` | Both two-lattice fitters, initialization, geometry, and comparison plots |
| `src/moire_cdw/calibration.py` | Independent calibration of detected or fitted vectors |
| `src/moire_cdw/tri.py` | Three-lattice detection, geometry, fitting, and plots |
| `src/moire_cdw/results.py` | Save/load fitting results for the calibration step |
| `src/moire_cdw/__init__.py` | Package entry point |

Fitting and tri each use a single file. Common fitting utilities live in
`fitting.py` and are reused by tri. The two fitting models remain separate
classes so their existing behavior is preserved.

## Examples

Run these notebooks in order; each saves a file that the next notebook can load
in a fresh kernel:

1. [Find peaks and vectors](examples/01_find_peaks_and_vectors.ipynb)
2. [Fit two lattice bases](examples/02_fitting.ipynb)
3. [Calibrate fitted vectors](examples/03_calibration.ipynb)

The [tri example](examples/tri/01_three_lattice.ipynb) is independent. It defaults
to a synthetic image with known seeds and supports experimental data with manual
or automatic peak indices.

The main example uses `data/Bi_1_total_0010.dm4`. Change paths and
settings in the parameter cells. Fitting defaults to `QUICK_RUN=True` (three
iterations per stage); set it to `False` for the original iteration schedule.
The short run demonstrates execution, not convergence.

The two-lattice detection example is tuned for this file with 4× mean binning:
`sigma=4`, `threshold=16000`, and `center_find_method="shape_center"` with a
50-pixel search radius. These settings reject weak satellite peaks and recover
two parent bases with lengths near 216 binned pixels and included angles near
60°. The notebook displays parent lattices first; enable `DRAW_MOIRE` to show
satellite predictions. Thresholds depend on the data and binning, and the shape
center setting assumes the direct beam is near the image midpoint. Library
algorithms and defaults are unchanged.

## Imports

```python
from moire_cdw import DiffractionMoire
from moire_cdw.fitting import (
    TwoLatticeMoireFitter, ExplicitMoireFitter,
    get_auto_init_from_moire, make_center_weight_mask,
    build_moire_from_vectors, plot_before_after_models,
)
from moire_cdw.calibration import calibration, calibration_from_vectors
from moire_cdw.tri import (
    DiffractionMoire as ThreeLatticeDetector,
    TripleLatticeMoireFitter, get_auto_init_from_moire_three,
)
from moire_cdw.results import save_fit_result, load_fit_result
```

`TwoLatticeMoireFitter` fits the parent lattices. `ExplicitMoireFitter` also enables
explicit moiré intensity in stage 3. Both retain the existing `build_from_auto_init`,
`fit_staged`, `summary`, and plotting interfaces. The optional
`fit_staged_with_stage_plots` helper is for the parent-only model.

All detection/fitting coordinates are `(row, col)` in **binned pixels**.
Calibration uses real-space d spacings and the binned reciprocal sampling; the
original algorithm uses `sampling[0]`. For sampling in `1/nm`, enter d spacings
in `nm`. The example's `0.276 nm` reference is inherited from the old notebook
and should be replaced for another material/reflection. Calibration returns
transformed vectors without resampling the image or changing the fit.

## Environment and checks

Python 3.11+ with NumPy, SciPy, Matplotlib, PyTorch, and QuantEM is required.
The notebooks import this checkout directly; for imports from other projects:

```sh
python -m pip install -e . --no-deps
```

Use an environment where QuantEM already works. The validated local interpreter
is `/Users/hu/Project/repos/quantem/.venv/bin/python` (Python 3.12.13, PyTorch 2.7.1).
The actual local QuantEM checkout was tested; its source metadata says 0.1.8,
while installed distribution metadata says 0.1.6. No environment was upgraded.

```sh
python -m pytest -q
python tools/run_examples.py
python tools/run_examples.py --in-place
python tools/run_examples.py --only 02_fitting.ipynb --explicit
```

The regression tests compare source definitions, numerical results, and staged
fits with the original notebooks. Use `--save-notebooks` to retain executed
review copies under `outputs/validation/`. By default only a compact validation
report is kept. The committed examples include executed outputs; use
`--in-place` to refresh them after changing settings. Numerical results are saved
separately under `outputs/two_lattice/` and `outputs/tri/`.

## Originals and retained behavior

`data/` contains the five unique DM4 files extracted from the original folders.
Byte-identical extra copies were removed only after SHA-256 verification.
The expanded `legacy/` folder and generated caches have been removed.

Original notebooks, source variants, and historical outputs are retained in the
single `tests/sources.zip` reference archive. Its manifest maps original files
to the archive or their new `data/` paths. `tests/sources.json` maps extracted
functions to original notebook cells. The regression tests read the archive
directly; there is no dependency on the old folder structure.
Original headers credit Colin Ophus and Guoliang Hu; no new license is assigned.

The numerical algorithms and defaults are retained, including the previous
refactor's small template-border clipping fix in explicit/tri rendering.
The new fitted-vector calibration adapter validates its inputs before calling
the unchanged calibration function.

The original tri detector can return fewer than three bases; the example reports
that case instead of silently inventing a third basis. The original
`real_peaks_intensity_range(return_peaks=True)` branch has an undefined `out_1`;
use its default `False` and read `moire.peaks` / `moire.I`, as the examples do.
