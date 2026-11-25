"""Execute examples in fresh kernels; optionally retain executed review copies.

Run from any directory using the Python environment that supplies QuantEM.
Use --in-place to save successful executions into the example notebooks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Example paths relative to examples/")
    parser.add_argument("--explicit", action="store_true", help="Run the two-lattice fitter in explicit-moire mode")
    parser.add_argument("--save-notebooks", action="store_true", help="Keep executed notebook copies under outputs/validation/")
    parser.add_argument("--in-place", action="store_true", help="Save successful runs, including figures, into examples/")
    args = parser.parse_args()
    if args.in_place and args.explicit:
        parser.error("Use --save-notebooks for the explicit variant to preserve the main example's settings.")
    names = args.only or ["01_find_peaks_and_vectors.ipynb", "02_fitting.ipynb",
                          "03_calibration.ipynb", "tri/01_three_lattice.ipynb"]
    output = ROOT / "outputs/validation" / ("explicit" if args.explicit else "default")
    if args.save_notebooks:
        output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        path = ROOT / "examples" / name
        notebook = nbformat.read(path, as_version=4)
        if args.explicit and name == "02_fitting.ipynb":
            for cell in notebook.cells:
                if "parameters" in cell.metadata.get("tags", []):
                    cell.source = cell.source.replace('MODEL = "parent"', 'MODEL = "explicit_moire"')
                    cell.source = cell.source.replace('two_lattice/fit.npz', 'two_lattice/fit_explicit.npz')
        start = time.monotonic()
        print(f"Running {name}", flush=True)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute(cleanup_kc=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        finally:
            if args.save_notebooks:
                nbformat.write(notebook, output / name.replace("/", "__"))
        if args.in_place:
            nbformat.write(notebook, path)
        elapsed = time.monotonic() - start
        records.append(dict(notebook=name, seconds=elapsed, status="passed",
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f"Passed {name} ({elapsed:.1f}s)", flush=True)
    report_path = ROOT / "outputs/validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["explicit" if args.explicit else "default"] = records
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
