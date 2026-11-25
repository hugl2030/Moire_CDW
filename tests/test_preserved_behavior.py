"""Regression checks against the archived notebook definitions, not rewritten formulas."""
import ast
import hashlib
import importlib
import inspect
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace, ModuleType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "tests/sources.json").read_text())
torch.set_num_threads(2)


def notebook_definitions(relative):
    """Execute imports and definitions only, excluding exploratory notebook cells."""
    namespace = {}
    notebook = json.loads(read_source(relative))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import)):
                nodes.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module not in ("superlattice", "superlattice_only_"):
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)
    return namespace


def read_source(relative):
    with zipfile.ZipFile(ROOT / "tests/sources.zip") as archive:
        return archive.read(relative).decode("utf-8")


def old_detector(relative):
    module = ModuleType("archived_detector")
    exec(compile(read_source(relative), relative, "exec"), module.__dict__)
    return module.DiffractionMoire


def assert_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (np.ndarray, list, tuple)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    elif a is None or isinstance(a, str):
        assert a == b
    else:
        assert a == pytest.approx(b, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("record", MANIFEST, ids=lambda x: x["module"] + "." + x["name"])
def test_extracted_definitions_unchanged(record):
    module = importlib.import_module(record["module"])
    definition = getattr(module, record["name"])
    node = ast.parse(inspect.getsource(definition)).body[0]
    notebook = json.loads(read_source(record["source"]))
    original_tree = ast.parse("".join(notebook["cells"][record["cell"]]["source"]))
    original = next(n for n in original_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and n.name == record["source_name"])
    original.name = record["name"]
    if record["name"] == "ExplicitMoireFitter":
        for child in ast.walk(original):
            if isinstance(child, ast.Name) and child.id == "build_moire_from_vectors":
                child.id = "_build_explicit_moire_from_vectors"
    if record["name"] in ("ExplicitMoireFitter", "TripleLatticeMoireFitter"):
        # The only algorithm-body edit: fix the template footprint at image edges.
        renderer = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                        and n.name == "_render_template_at_points")
        corrections = {"r1": "min(self.shape[0], r - cr + th)",
                       "c1": "min(self.shape[1], c - cc + tw)"}
        for statement in ast.walk(renderer):
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in corrections:
                    statement.value = ast.parse(corrections[name], mode="eval").body
    # Parse both sides with this interpreter: AST fields differ by Python version.
    assert ast.dump(node) == ast.dump(original)


@pytest.mark.parametrize("old,new", [
    ("Double_basis_vectors/superlattice.py", "src/moire_cdw/superlattice.py"),
    ("tri_fitting/superlattice_only_.py", "src/moire_cdw/tri.py"),
])
def test_detectors_unchanged(old, new):
    trees = [ast.parse(read_source(old)), ast.parse((ROOT/new).read_text())]
    classes = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffractionMoire")
               for tree in trees]
    for cls in classes:
        for node in ast.walk(cls):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node) is not None:
                doc = node.body[0].value
                doc.value = "\n".join(line.rstrip() for line in doc.value.split("\n"))
    assert ast.dump(classes[0]) == ast.dump(classes[1])


def test_original_files_preserved():
    with zipfile.ZipFile(ROOT / "tests/sources.zip") as archive:
        records = json.loads(archive.read("manifest.json"))
        for entry in records:
            if entry["kind"] == "removed_cache":
                continue
            content = ((ROOT / entry["location"]).read_bytes() if entry["kind"] == "data"
                       else archive.read(entry["location"]))
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert len({r["location"] for r in records if r["kind"] == "data"}) == 5


@pytest.fixture
def pattern():
    origin = np.array([48., 48.])
    vectors = np.array([(0., 24.), (24., 0.), (12., 20.), (-20., 12.), (20., 5.), (-5., 20.)])
    rr, cc = np.indices((96, 96))
    image = np.ones((96, 96), dtype=np.float32)
    for point in np.vstack([origin, origin + vectors, origin - vectors]):
        image += 100 * np.exp(-((rr-point[0])**2 + (cc-point[1])**2) / 4)
    return image, origin, vectors


@pytest.mark.parametrize("legacy,module", [
    ("Double_basis_vectors/superlattice.py", "moire_cdw.superlattice"),
    ("tri_fitting/superlattice_only_.py", "moire_cdw.tri"),
])
def test_peak_finding_and_intensity_parity(pattern, legacy, module):
    image, _, _ = pattern
    old = old_detector(legacy).from_data(image.copy())
    new = importlib.import_module(module).DiffractionMoire.from_data(image.copy())
    for obj in (old, new):
        obj.find_peaks(threshold=10, min_distance_px=3, refine_radius_px=3, plot_result=False)
        obj.real_peaks_intensity_range(distance=3, plot_result=False)
    assert_equal(old.peaks, new.peaks)
    assert_equal(old.I, new.I)


@pytest.mark.parametrize("source,module,class_name", [
    ("Fitting model/moire_auto_init_plus_fitting_with_comparison.ipynb",
     "moire_cdw.fitting", "TwoLatticeMoireFitter"),
    ("Fitting model/moire_auto_init_plus_fitting_with_trained_moire-Copy1.ipynb",
     "moire_cdw.fitting", "ExplicitMoireFitter"),
    ("tri_fitting/1.ipynb", "moire_cdw.tri", "TripleLatticeMoireFitter"),
])
def test_fitting_stages_match_archived_notebooks(pattern, source, module, class_name):
    image, origin, vectors = pattern
    old_name = "TwoLatticeMoireFitter" if class_name == "ExplicitMoireFitter" else class_name
    old_cls = notebook_definitions(source)[old_name]
    new_cls = getattr(importlib.import_module(module), class_name)
    kwargs = dict(origin_rc=origin, u1_rc=vectors[0], v1_rc=vectors[1],
                  u2_rc=vectors[2], v2_rc=vectors[3], peak_seed_rc=origin+vectors[:4],
                  lattice_order=2, center_radius_px=3, spot_radius_px=2)
    if class_name == "ExplicitMoireFitter":
        kwargs.update(moire_parent_maxlen_px=24, moire_max_tile=3)
    if class_name == "TripleLatticeMoireFitter":
        kwargs.update(u3_rc=vectors[4], v3_rc=vectors[5], moire_parent_maxlen_px=24, max_tile=3)
    outputs = []
    for cls in (old_cls, new_cls):
        torch.manual_seed(0)
        fitter = cls(image.copy())
        fitter.build_from_auto_init(**kwargs)
        before = fitter.render_numpy()
        fitter.fit_staged(stage1_steps=2, stage2_steps=2, stage3_steps=2)
        outputs.append((before, fitter.render_numpy(), fitter.summary(), fitter.loss_history))
    for a, b in zip(*outputs):
        assert_equal(a, b)
    assert np.isfinite(outputs[1][1]).all()
    assert np.isfinite(outputs[1][3]).all()


def test_post_fit_calibration_matches_original(pattern, tmp_path):
    from moire_cdw.calibration import calibration_from_vectors
    from moire_cdw.results import save_fit_result, load_fit_result
    _, origin, vectors = pattern
    summary = dict(origin_rc=origin, u1_rc=vectors[0], v1_rc=vectors[1],
                   u2_rc=vectors[2], v2_rc=vectors[3], final_loss=0.1,
                   trained_moire_points_rc=None)
    path = tmp_path / "fit.npz"
    save_fit_result(path, summary, sampling=[0.02, 0.02], units=["1/nm", "1/nm"],
                    bin_factor=4, initial_summary=summary, loss_history=[1., 0.1])
    loaded = load_fit_result(path)
    assert_equal(summary, loaded["summary"])
    assert loaded["metadata"]["bin_factor"] == 4
    old = notebook_definitions("Calibration/moire_test_03.ipynb")["calibration"]
    reference = old(np.vstack([origin, origin+vectors[:4]]), [0, 1, 2, 3, 4],
                    SimpleNamespace(sampling=[0.02, 0.02]), return_info=True, plot_vectors=False)
    selected = {key: loaded["summary"][key] for key in ("origin_rc", "u1_rc", "v1_rc", "u2_rc", "v2_rc")}
    actual = calibration_from_vectors(**selected, sampling=loaded["sampling"], plot_vectors=False)
    assert_equal(reference, actual)
    assert actual["angle_after_1"] == pytest.approx(actual["angle_target"])
    np.testing.assert_allclose(actual["A"] @ vectors[0], actual["u1_ref"])


def test_calibration_missing_reference_and_singular_basis(pattern):
    from moire_cdw.calibration import calibration_from_vectors
    _, origin, v = pattern
    result = calibration_from_vectors(origin, *v[:4], sampling=0.02, u1_calibrate=None,
                                       plot_vectors=False)
    assert np.isscalar(result)
    with pytest.raises(ValueError, match="singular"):
        calibration_from_vectors(origin, v[0], 2*v[0], v[2], v[3], sampling=0.02,
                                 plot_vectors=False)
    with pytest.raises(ValueError, match="sampling"):
        calibration_from_vectors(origin, *v[:4], sampling=0, plot_vectors=False)


@pytest.mark.parametrize("module,class_name", [
    ("moire_cdw.fitting", "ExplicitMoireFitter"),
    ("moire_cdw.tri", "TripleLatticeMoireFitter"),
])
def test_template_edges_and_gradients(module, class_name):
    cls = getattr(importlib.import_module(module), class_name)
    fitter = cls(np.ones((12, 14), dtype=np.float32))
    template = torch.arange(1., 26.).reshape(5, 5).requires_grad_()
    scale = torch.tensor(0.3, requires_grad=True)
    points = np.array([(0, 0), (0, 13), (11, 0), (11, 13), (-1, 5), (6, 6)])
    actual = fitter._render_template_at_points(template, points, scale)
    expected = np.zeros((12, 14), dtype=np.float32)
    # Independent pixel-by-pixel reference handles corners and partial templates.
    for row, col in points:
        for tr in range(5):
            for tc in range(5):
                r, c = row + tr - 2, col + tc - 2
                if 0 <= r < 12 and 0 <= c < 14:
                    expected[r, c] += 0.3 * (tr*5+tc+1) / 25
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    assert torch.isfinite(template.grad).all()
    assert torch.isfinite(scale.grad)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
