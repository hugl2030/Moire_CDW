"""Portable fit-to-calibration handoff (NPZ arrays + JSON, no pickle)."""
import json
from pathlib import Path

import numpy as np

VECTOR_KEYS = ("origin_rc", "u1_rc", "v1_rc", "u2_rc", "v2_rc")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_fit_result(path, summary, *, sampling, units, source_file="",
                    bin_factor=1, image_shape=None, loss_history=(),
                    initial_summary=None, fit_settings=None):
    """Save vectors in binned pixels with their sampling and fitting settings.

    Replaces ``path`` if it already exists. No image/raw-data copy is made.
    A triple-lattice summary may also contain u3_rc/v3_rc.
    """
    path = Path(path)
    arrays = {key: np.asarray(summary[key], dtype=float) for key in VECTOR_KEYS}
    for key in ("u3_rc", "v3_rc", "trained_moire_points_rc"):
        if summary.get(key) is not None:
            arrays[key] = np.asarray(summary[key], dtype=float)
    if any(arrays[k].shape != (2,) for k in VECTOR_KEYS):
        raise ValueError("origin/vectors must be (row, col) pairs")
    arrays["sampling"] = np.asarray(sampling, dtype=float)
    arrays["loss_history"] = np.asarray(loss_history, dtype=float)
    metadata = dict(
        format_version=1, coordinates="row_col_binned_pixels",
        units=list(units), source_file=str(source_file), bin_factor=int(bin_factor),
        image_shape=image_shape, initial_summary=initial_summary,
        fit_settings=fit_settings,
        summary={k: v for k, v in summary.items() if k not in arrays},
    )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, default=_json_value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_fit_result(path):
    """Return summary, sampling, units, loss_history and metadata without pickle."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported fit-result format")
        summary = dict(metadata["summary"])
        for key in (*VECTOR_KEYS, "u3_rc", "v3_rc", "trained_moire_points_rc"):
            if key in data:
                summary[key] = data[key].copy()
        return dict(summary=summary, sampling=data["sampling"].copy(),
                    units=metadata["units"], loss_history=data["loss_history"].copy(),
                    metadata=metadata)
