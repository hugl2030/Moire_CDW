"""Three-lattice peak detection, geometry, fitting, and plots."""
from __future__ import annotations

from os import PathLike
from typing import Any, Self
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, KDTree
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
import matplotlib.pyplot as plt
import torch
from quantem.core.fitting.base import OriginND, RenderContext, AdditiveRenderModel
from quantem.core.fitting.diffraction import DiskTemplate, SyntheticDiskLattice
from quantem.core.fitting.background import DCBackground, GaussianBackground
from .fitting import robust_normalize, extract_patch, make_center_weight_mask, plot_before_after_models

# Three-lattice detector

class DiffractionMoire(AutoSerialize):
    _token = object()

    def __init__(
        self,
        diffraction: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use DiffractionMoire.from_data() or .from_file() to instantiate this class.")
        self._diffraction = diffraction
        self.peaks = np.zeros((0, 2), dtype=float)

    @classmethod
    def from_file(
        cls,
        file_path: str | PathLike,
        file_type: str | None = None,
    ) -> Self:
        diffraction = Dataset2d.from_file(str(file_path), file_type=file_type)
        return cls.from_data(diffraction)

    @classmethod
    def from_data(
        cls,
        diffraction: Dataset2d | NDArray,
        *,
        name: str | None = None,
        origin: NDArray | float | int | None = None,
        sampling: NDArray | float | int | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        if isinstance(diffraction, Dataset2d):
            ds = diffraction
            if name is not None:
                ds.name = name
        else:
            arr = ensure_valid_array(diffraction, ndim=2)
            ds = Dataset2d.from_array(
                arr,
                name=name if name is not None else "diffraction",
                origin=origin if origin is not None else np.zeros(2),
                sampling=sampling if sampling is not None else np.ones(2),
                units=units if units is not None else ["pixels"] * 2,
                signal_units=signal_units,
            )

        return cls(
            diffraction=ds,
            _token=cls._token,
        )

    @property
    def diffraction(self) -> Dataset2d:
        return self._diffraction

    def find_diffraction_center_robust(self, peaks, tolerance=3.0, max_iters=20000):
        """
        Robustly back-calculate diffraction center using RANSAC to ignore noise.
        """
        peaks = np.asarray(peaks, dtype=float)
        num_peaks = len(peaks)
        if num_peaks < 2:
            return np.mean(peaks, axis=0) if num_peaks > 0 else np.zeros(2)

        tree = cKDTree(peaks)

        best_center = np.mean(peaks, axis=0)
        max_inliers = -1

        for _ in range(max_iters):
            idx1, idx2 = np.random.choice(num_peaks, 2, replace=False)
            candidate_center = (peaks[idx1] + peaks[idx2]) / 2

            flipped = 2 * candidate_center - peaks
            dists, _ = tree.query(flipped, k=1)

            inliers = np.sum(dists < tolerance)

            if inliers > max_inliers:
                max_inliers = inliers
                best_center = candidate_center

        flipped_best = 2 * best_center - peaks
        dists, indices = tree.query(flipped_best, k=1)
        mask = dists < tolerance

        valid_pairs_midpoints = (peaks[mask] + peaks[indices[mask]]) / 2
        refined_center = np.mean(valid_pairs_midpoints, axis=0)

        print(f"Result: find {max_inliers} symmetric point pairs, final center: {refined_center}")
        return refined_center

    def find_peaks(
        self,
        *,
        threshold: float | None = None,
        min_distance_px: float | None = None,
        max_peaks: int | None = None,
        sigma_range: tuple[float, float] = (3.0, 1.0), # Sigma scan range
        sigma_step: float = 0.1,
        mask_diffraction: NDArray | None = None,
        refine_subpixel: bool = True,
        refine_radius_px: int = 4,
        refine_maxfev: int = 500,
        plot_result: bool = True,
        plot_radius_min_px: float = 2.0,
        plot_radius_max_px: float = 10.0,
        plot_radius_power: float = 0.5,
        plot_edgecolor: str = "r",
        plot_linewidth: float = 1.0,
        plot_alpha: float = 0.5,
        plot_labels: bool = False,
        plot_label_color: Any = (0, 0.9, 1.0),
        plot_label_size: float = 6.0,
        plot_label_alpha: float = 0.9,
        plot_label_offset_px: tuple[float, float] = (0.0, -12.0),
        plot_label_bbox: bool = True,
        plot_label_bbox_facecolor: Any = (0.0, 0.0, 0.0, 0.35),
        plot_label_bbox_edgecolor: Any = "none",
        plot_label_bbox_pad: float = 0.15,
        plot_label_top_k: int | None = None,
        return_peaks: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        im = np.asarray(self.diffraction.array)
        if im.ndim != 2:
            raise ValueError("DiffractionMoire.find_peaks requires a 2D diffraction array.")

        # Initialize the result container.
        all_detected_peaks = np.zeros((0, 2), dtype=float)

        # Scan in decreasing sigma order.
        # Decrease from 3.0 to 1.0 in steps of 0.1.
        sigmas = np.arange(sigma_range[0], sigma_range[1] - 1e-5, -sigma_step)

        for s in sigmas:
            im_s = gaussian_filter(im, sigma=float(s)) if float(s) > 0 else im

            # Compute the threshold at each sigma unless supplied explicitly.
            if threshold is None:
                mu = float(np.nanmean(im_s))
                sig = float(np.nanstd(im_s))
                curr_threshold = mu + 5.0 * sig
            else:
                curr_threshold = float(threshold)

            # Adjust the non-maximum suppression window to min_distance_px.
            win = max(3, int(np.ceil(min_distance_px) * 2 + 1))
            mx = maximum_filter(im_s, size=win, mode="nearest")
            cand_mask = (im_s == mx) & (im_s >= curr_threshold)

            if mask_diffraction is not None:
                cand_mask &= np.asarray(mask_diffraction).astype(bool)

            new_peaks = np.argwhere(cand_mask).astype(float)

            if new_peaks.size == 0:
                continue

            # Merge spatially overlapping detections.
            if all_detected_peaks.size == 0:
                all_detected_peaks = new_peaks
            else:
                from scipy.spatial import cKDTree
                tree = cKDTree(all_detected_peaks)
                # Find the nearest distance from each new point to the existing set.
                dists, _ = tree.query(new_peaks, k=1)

                # Only points farther than min_distance_px are considered new.
                # Keep the larger-sigma coordinates when a point shifts during the scan.
                unique_mask = dists > min_distance_px
                if np.any(unique_mask):
                    all_detected_peaks = np.vstack([all_detected_peaks, new_peaks[unique_mask]])

        # Sort the merged results.
        # Peaks now contain merged detections across all sigma levels.
        # Reorder by intensity in the original image.
        peak_I = im[all_detected_peaks[:, 0].astype(int), all_detected_peaks[:, 1].astype(int)].astype(float)
        order = np.argsort(-peak_I)
        self.peaks = all_detected_peaks[order]
        self.I = peak_I[order]
        peaks_rc = self.peaks

        if max_peaks is not None and self.peaks.shape[0] > int(max_peaks):
            self.peaks = self.peaks[: int(max_peaks)]
            self.I = self.I[: int(max_peaks)]

        # Initialize the output dictionary.
        out = {
            "peaks": self.peaks,
            "I": self.I,
            "threshold": threshold, # Record the user-specified threshold.
            "sigma_range": sigma_range,
        }

        if refine_subpixel and peaks_rc.shape[0] > 0:
            r = int(refine_radius_px)
            if r < 1:
                raise ValueError("refine_radius_px must be >= 1.")

            def _gauss2d(coords: NDArray, A: float, row0: float, col0: float, s: float, c: float) -> NDArray:
                rr, cc = coords
                return A * np.exp(-((rr - row0) ** 2 + (cc - col0) ** 2) / (2.0 * s * s)) + c

            peaks_rc_ref = peaks_rc.copy()
            I_ref = peak_I.copy()
            fit_params = np.full((peaks_rc.shape[0], 5), np.nan, dtype=float)
            fit_success = np.zeros((peaks_rc.shape[0],), dtype=bool)

            for k in range(peaks_rc.shape[0]):
                row0_i = float(peaks_rc[k, 0])
                col0_i = float(peaks_rc[k, 1])
                row_i = int(np.round(row0_i))
                col_i = int(np.round(col0_i))

                r0 = max(0, row_i - r)
                r1 = min(im_s.shape[0], row_i + r + 1)
                c0 = max(0, col_i - r)
                c1 = min(im_s.shape[1], col_i + r + 1)
                if (r1 - r0) < 3 or (c1 - c0) < 3:
                    continue

                patch = im_s[r0:r1, c0:c1].astype(float, copy=False)
                rr, cc = np.mgrid[r0:r1, c0:c1]
                xdata = np.vstack([rr.ravel(), cc.ravel()])
                ydata = patch.ravel()

                c_init = float(np.median(ydata))
                A_init = float(np.max(ydata) - c_init)
                if not np.isfinite(A_init) or A_init <= 0:
                    continue
                s_init = max(0.75, float(r) / 2.0)

                row_min = float(r0)
                row_max = float(r1 - 1)
                col_min = float(c0)
                col_max = float(c1 - 1)

                p0 = (A_init, row0_i, col0_i, s_init, c_init)
                lb = (0.0, row_min, col_min, 0.5, float(np.min(ydata) - abs(A_init)))
                ub = (
                    float(np.inf),
                    row_max,
                    col_max,
                    float(max(0.75, 2.0 * r)),
                    float(np.max(ydata) + abs(A_init)),
                )

                try:
                    popt, _ = curve_fit(
                        _gauss2d,
                        xdata,
                        ydata,
                        p0=p0,
                        bounds=(lb, ub),
                        maxfev=int(refine_maxfev),
                    )
                except Exception:
                    continue

                A_fit, row_fit, col_fit, s_fit, c_fit = (
                    float(popt[0]),
                    float(popt[1]),
                    float(popt[2]),
                    float(popt[3]),
                    float(popt[4]),
                )
                if not (np.isfinite(row_fit) and np.isfinite(col_fit) and np.isfinite(s_fit) and s_fit > 0):
                    continue

                peaks_rc_ref[k, 0] = row_fit
                peaks_rc_ref[k, 1] = col_fit
                I_ref[k] = max(A_fit + c_fit, float(im[row_i, col_i]))
                fit_params[k, :] = np.array([A_fit, row_fit, col_fit, s_fit, c_fit], dtype=float)
                fit_success[k] = True

            order_ref = np.argsort(-I_ref)
            peaks_rc_ref = peaks_rc_ref[order_ref]
            I_ref = I_ref[order_ref]
            fit_params = fit_params[order_ref]
            fit_success = fit_success[order_ref]

            out["peaks_refined"] = peaks_rc_ref
            out["I_refined"] = I_ref
            out["gaussian_fit_params"] = fit_params
            out["gaussian_fit_success"] = fit_success
            out["refine_radius_px"] = int(refine_radius_px)

            self.peaks = peaks_rc_ref
            self.I = I_ref
            out["peaks"] = peaks_rc_ref
            out["I"] = I_ref

        if plot_result:
            from matplotlib.patches import Circle

            fig, ax = show_2d(im_s, **kwargs)
            ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

            peaks_plot = np.asarray(out["peaks"], dtype=float)
            I_plot = np.asarray(out["I"], dtype=float)

            if peaks_plot.size:
                I_min = float(np.nanmin(I_plot)) if I_plot.size else 0.0
                I_max = float(np.nanmax(I_plot)) if I_plot.size else 1.0
                denom = I_max - I_min
                if not np.isfinite(denom) or denom <= 0:
                    t = np.ones_like(I_plot, dtype=float)
                else:
                    t = np.clip((I_plot - I_min) / denom, 0.0, 1.0)

                t = np.power(t, float(plot_radius_power))
                radii = float(plot_radius_min_px) + (float(plot_radius_max_px) - float(plot_radius_min_px)) * t

                for (row, col), rad in zip(peaks_plot, radii):
                    ax0.add_patch(
                        Circle(
                            (col, row),
                            radius=float(rad),
                            facecolor="none",
                            edgecolor=plot_edgecolor,
                            linewidth=float(plot_linewidth),
                            alpha=float(plot_alpha),
                        )
                    )

                if plot_labels:
                    dcol, drow = float(plot_label_offset_px[0]), float(plot_label_offset_px[1])
                    bbox_kw = None
                    if plot_label_bbox:
                        bbox_kw = dict(
                            boxstyle=f"round,pad={float(plot_label_bbox_pad)}",
                            facecolor=plot_label_bbox_facecolor,
                            edgecolor=plot_label_bbox_edgecolor,
                        )

                    nlab = peaks_plot.shape[0]
                    if plot_label_top_k is not None:
                        nlab = min(nlab, int(plot_label_top_k))

                    for idx in range(nlab):
                        row, col = float(peaks_plot[idx, 0]), float(peaks_plot[idx, 1])
                        ax0.text(
                            float(col + dcol),
                            float(row + drow),
                            str(idx),
                            color=plot_label_color,
                            fontsize=float(plot_label_size),
                            alpha=float(plot_label_alpha),
                            ha="center",
                            va="center",
                            bbox=bbox_kw,
                        )

            ax0.figure.canvas.draw_idle()

        if return_peaks:
            return out

    def real_peaks_intensity_range(
        self,
        distance = 4.0,
        sigma = 1,
        plot_result: bool = True,
        plot_radius_min_px: float = 2.0,
        plot_radius_max_px: float = 10.0,
        plot_radius_power: float = 0.5,
        plot_edgecolor: str = "r",
        plot_linewidth: float = 1.0,
        plot_alpha: float = 0.5,
        plot_labels: bool = False,
        plot_label_color: Any = (0, 0.9, 1.0),
        plot_label_size: float = 6.0,
        plot_label_alpha: float = 0.9,
        plot_label_offset_px: tuple[float, float] = (0.0, -12.0),
        plot_label_bbox: bool = True,
        plot_label_bbox_facecolor: Any = (0.0, 0.0, 0.0, 0.35),
        plot_label_bbox_edgecolor: Any = "none",
        plot_label_bbox_pad: float = 0.15,
        plot_label_top_k: int | None = None,
        return_peaks: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        im = np.asarray(self.diffraction.array)
        im_s = gaussian_filter(im, sigma=float(sigma)) if float(sigma) > 0 else im
        peaks = np.asarray(self.peaks, dtype=float)

        H, W = im.shape
        I = []

        dist = int(np.ceil(distance if distance is not None else 4.0))
        if dist <= 0:
            raise ValueError("distance must be > 0.")

        for k in range(peaks.shape[0]):

            r = int(np.rint(peaks[k, 0]))
            c = int(np.rint(peaks[k, 1]))

            row0 = int(max(r - dist, 0))
            row1 = int(min(r + dist + 1, H))
            col0 = int(max(c - dist, 0))
            col1 = int(min(c + dist + 1, W))

            patch = im[row0:row1, col0:col1]
            I.append(float(patch.max()))

        I_arr = np.asarray(I)
        order_ref = np.argsort(-I_arr)
        peaks_rc_ref = peaks[order_ref]
        I_ref = I_arr[order_ref]
        out: dict[str, Any] = {
            "peaks": peaks_rc_ref,
            "I": I_ref,
        }
        self.peaks = peaks_rc_ref
        self.I = I_ref

        if plot_result:
            from matplotlib.patches import Circle

            fig, ax = show_2d(im_s, **kwargs)
            ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

            peaks_plot = np.asarray(out["peaks"], dtype=float)
            I_plot = np.asarray(out["I"], dtype=float)

            if peaks_plot.size:
                I_min = float(np.nanmin(I_plot)) if I_plot.size else 0.0
                I_max = float(np.nanmax(I_plot)) if I_plot.size else 1.0
                denom = I_max - I_min
                if not np.isfinite(denom) or denom <= 0:
                    t = np.ones_like(I_plot, dtype=float)
                else:
                    t = np.clip((I_plot - I_min) / denom, 0.0, 1.0)

                t = np.power(t, float(plot_radius_power))
                radii = float(plot_radius_min_px) + (float(plot_radius_max_px) - float(plot_radius_min_px)) * t

                for (row, col), rad in zip(peaks_plot, radii):
                    ax0.add_patch(
                        Circle(
                            (col, row),
                            radius=float(rad),
                            facecolor="none",
                            edgecolor=plot_edgecolor,
                            linewidth=float(plot_linewidth),
                            alpha=float(plot_alpha),
                        )
                    )

                if plot_labels:
                    dcol, drow = float(plot_label_offset_px[0]), float(plot_label_offset_px[1])
                    bbox_kw = None
                    if plot_label_bbox:
                        bbox_kw = dict(
                            boxstyle=f"round,pad={float(plot_label_bbox_pad)}",
                            facecolor=plot_label_bbox_facecolor,
                            edgecolor=plot_label_bbox_edgecolor,
                        )

                    nlab = peaks_plot.shape[0]
                    if plot_label_top_k is not None:
                        nlab = min(nlab, int(plot_label_top_k))

                    for idx in range(nlab):
                        row, col = float(peaks_plot[idx, 0]), float(peaks_plot[idx, 1])
                        ax0.text(
                            float(col + dcol),
                            float(row + drow),
                            str(idx),
                            color=plot_label_color,
                            fontsize=float(plot_label_size),
                            alpha=float(plot_label_alpha),
                            ha="center",
                            va="center",
                            bbox=bbox_kw,
                        )

            ax0.figure.canvas.draw_idle()

        if return_peaks:
            return out_1


    def auto_find_lattice_vectors(
        self,
        peaks_number: int = 150,
        num: int = 3,
    ) -> list:
        """
        Automatically identify up to `num` lattice-vector pairs.

        Notes
        -----
        This version keeps the original algorithm and selection logic as intact as possible.
        The main changes are:
        1. code cleanup / clearer structure
        2. small safe speedups (KDTree reuse + per-pair caching)
        3. a couple of defensive fixes to avoid accidental runtime errors
        """
        peaks = np.asarray(self.peaks, dtype=float)
        intensity = np.asarray(self.I, dtype=float)
        im = np.asarray(self.diffraction.array)
        nrow, ncol = int(im.shape[0]), int(im.shape[1])

        # --- 1) Locate center / origin peak ---
        true_center = self.find_diffraction_center_robust(peaks)
        dists_to_true = np.linalg.norm(peaks - true_center, axis=1)

        search_radius = 10.0
        mask_near = dists_to_true < search_radius

        if np.any(mask_near):
            near_indices = np.where(mask_near)[0]
            highest_i_sub_idx = np.argmax(intensity[near_indices])
            center_peak_index = near_indices[highest_i_sub_idx]

            # Keep original behavior: refine that peak to the robust center.
            peaks[center_peak_index] = true_center
            print(
                f"Physical peak found near center, index: {center_peak_index}. "
                "Coordinates refined to True_center."
            )
        else:
            peaks = np.vstack([peaks, true_center.reshape(1, 2)])
            intensity = np.append(intensity, 0.0)
            center_peak_index = len(peaks) - 1
            print(
                f"No physical peak found near center. Ideal center added at index: "
                f"{center_peak_index} with intensity 0."
            )

        origin = true_center
        max_radius = min(nrow, ncol) // 2

        # --- 2) Candidate generation ---
        def generate_combination(peaks_number, origin, filtered_peaks, filtered_intensity):
            # peaks_number is intentionally kept here to preserve the original signature/behavior.
            idx_range = np.arange(len(filtered_intensity))
            multiple_intensity = (idx_range + 1) * filtered_intensity

            aa, bb = np.meshgrid(idx_range, idx_range)
            combinations = np.vstack([bb.ravel(), aa.ravel()]).T  # all pairs

            p_a = filtered_peaks[combinations[:, 0]].astype(float, copy=False)
            p_b = filtered_peaks[combinations[:, 1]].astype(float, copy=False)

            vec_oa = p_a - origin
            vec_ob = p_b - origin
            dot_products = np.sum(vec_oa * vec_ob, axis=1)

            norms_oa = np.linalg.norm(vec_oa, axis=1)
            norms_ob = np.linalg.norm(vec_ob, axis=1)
            cos_thetas = np.clip(dot_products / (norms_oa * norms_ob + 1e-9), -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_thetas))

            mask = (
                (angles >= 40)
                & (angles <= 140)
                & (norms_oa > 0)
                & (norms_ob > 0)
            )

            select_combinations = combinations[mask]
            return select_combinations, multiple_intensity

        # --- 3) Lattice point helpers ---
        def _default_max_tile(u, v) -> int:
            du = float(np.linalg.norm(u))
            dv = float(np.linalg.norm(v))
            dmin = min(du, dv) if (du > 0 and dv > 0) else max(du, dv)
            if not np.isfinite(dmin) or dmin <= 0:
                return 10
            diag = float(np.hypot(nrow, ncol))
            mt = int(np.ceil(diag / dmin)) + 2
            return int(np.clip(mt, 3, 200))

        def _tile_points_bounded(r0, u, v, mt):
            if mt == 0:
                return r0[None, :].copy()

            a = np.arange(-mt, mt + 1, dtype=float)
            b = np.arange(-mt, mt + 1, dtype=float)
            aa, bb = np.meshgrid(a, b, indexing="ij")

            pts = (
                r0[None, None, :]
                + aa[..., None] * u[None, None, :]
                + bb[..., None] * v[None, None, :]
            )
            pts = pts.reshape(-1, 2)

            keep = (
                (pts[:, 0] >= 0.0)
                & (pts[:, 0] <= (nrow - 1))
                & (pts[:, 1] >= 0.0)
                & (pts[:, 1] <= (ncol - 1))
            )
            return pts[keep]

        try:
            from scipy.spatial import cKDTree as _KDTree
        except Exception:
            from scipy.spatial import KDTree as _KDTree

        def generate_two_lattice_moire(
            r0,
            u1,
            v1,
            u2,
            v2,
            nrow,
            ncol,
            maxlen=250,
            r_excl=2.0,
            decimals=3,
        ):
            pts_moire = None

            def _disp_set(u, v, maxlen):
                dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
                if not np.isfinite(dmin) or dmin <= 0:
                    raise ValueError("Invalid lattice vectors for moire generation.")

                mt = int(np.ceil(maxlen / dmin)) + 1
                mt = int(np.clip(mt, 1, 200))
                a = np.arange(-mt, mt + 1, dtype=float)
                b = np.arange(-mt, mt + 1, dtype=float)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                aa = aa.ravel()
                bb = bb.ravel()

                disp = aa[:, None] * u[None, :] + bb[:, None] * v[None, :]
                dn = np.sqrt(np.sum(disp**2, axis=1))
                keep = dn <= maxlen + 1e-9

                aa = aa[keep]
                bb = bb[keep]
                disp = disp[keep]
                return disp, np.stack([aa, bb], axis=1)

            disp1, ab1 = _disp_set(u1, v1, maxlen)
            disp2, ab2 = _disp_set(u2, v2, maxlen)

            if disp1.size and disp2.size:
                n1 = int(disp1.shape[0])
                n2 = int(disp2.shape[0])
                if n1 * n2 > 2_000_000:
                    raise ValueError(
                        f"Moire candidate count too large (N1*N2={n1*n2}). "
                        "Reduce moire_parent_maxlen_px to limit the search space."
                    )

                pts = r0[None, None, :] + disp1[:, None, :] + disp2[None, :, :]
                pts = pts.reshape(-1, 2)

                keep_combo = ~(
                    ((ab1[:, 0] == 0) & (ab1[:, 1] == 0))[:, None]
                    | ((ab2[:, 0] == 0) & (ab2[:, 1] == 0))[None, :]
                ).ravel()
                pts = pts[keep_combo]

                keep_bounds = (
                    (pts[:, 0] >= 0.0)
                    & (pts[:, 0] <= (nrow - 1))
                    & (pts[:, 1] >= 0.0)
                    & (pts[:, 1] <= (ncol - 1))
                )
                pts = pts[keep_bounds]

                if pts.size:
                    pts = np.unique(np.round(pts, int(decimals)), axis=0)

                    if r_excl > 0:
                        parent_pts = np.vstack((r0[None, :] + disp1, r0[None, :] + disp2))
                        parent_keep = (
                            (parent_pts[:, 0] >= 0.0)
                            & (parent_pts[:, 0] <= (nrow - 1))
                            & (parent_pts[:, 1] >= 0.0)
                            & (parent_pts[:, 1] <= (ncol - 1))
                        )
                        parent_pts = parent_pts[parent_keep]

                        if parent_pts.size:
                            try:
                                tree = _KDTree(parent_pts)
                                try:
                                    d, _ = tree.query(pts, k=1, workers=-1)
                                except TypeError:
                                    d, _ = tree.query(pts, k=1)
                                pts = pts[d >= r_excl]
                            except Exception:
                                r2 = r_excl * r_excl
                                keep = np.ones(pts.shape[0], dtype=bool)
                                for i in range(pts.shape[0]):
                                    d2 = np.sum((parent_pts - pts[i]) ** 2, axis=1)
                                    if np.any(d2 < r2):
                                        keep[i] = False
                                pts = pts[keep]

                    pts_moire = pts if pts.size else None

            return pts_moire

        final_basis_results = []
        filtered_peaks = peaks

        # --- 4) Loop over mask radius and rank candidates ---
        for current_mask_r in range(50, max_radius, 10):
            final_basis_results = []

            dists = np.linalg.norm(peaks - origin, axis=1)
            keep_mask = dists >= current_mask_r
            keep_mask[center_peak_index] = True

            filtered_peaks = peaks[keep_mask]
            filtered_intensity = intensity[keep_mask]
            print(len(filtered_peaks))

            select_combinations, multiple_intensity = generate_combination(
                peaks_number,
                origin,
                filtered_peaks,
                filtered_intensity,
            )

            if select_combinations.size == 0:
                continue

            filtered_tree = _KDTree(filtered_peaks)
            pair_cache = {}

            def _find_nearby_peaks_fast(target_points, radius=2):
                results = []
                for pt in target_points:
                    indices = filtered_tree.query_ball_point(pt, r=radius)
                    if indices:
                        results.append(indices[0])
                    else:
                        results.append(False)
                return results

            def _find_index_in_peaks(u, v):
                key = (int(u), int(v))
                if key in pair_cache:
                    return pair_cache[key]

                u_k = filtered_peaks[u].astype(float, copy=False) - origin
                v_k = filtered_peaks[v].astype(float, copy=False) - origin
                mt = _default_max_tile(u_k, v_k)
                pts = _tile_points_bounded(origin, u_k, v_k, mt)
                index = _find_nearby_peaks_fast(pts, radius=2)
                unique_indices = [i for i in index if i is not False]

                pair_cache[key] = unique_indices
                return unique_indices

            total_intensity = np.empty(select_combinations.shape[0], dtype=float)
            for k, (u, v) in enumerate(select_combinations):
                unique_indices = _find_index_in_peaks(u, v)
                total_intensity[k] = np.sum(multiple_intensity[unique_indices])

            order_ref = np.argsort(-total_intensity)
            select_combinations_ref = select_combinations[order_ref]

            print(select_combinations_ref)
            print(len(select_combinations_ref))

            if select_combinations_ref.size == 0:
                continue

            u_1, v_1 = select_combinations_ref[0]
            final_basis_results.append([u_1, v_1])

            k_indices = _find_index_in_peaks(u_1, v_1)
            forbidden_indices = set(k_indices)
            forbidden_indices.discard(center_peak_index)

            mask = np.ones(len(select_combinations_ref), dtype=bool)
            for j, (u, v) in enumerate(select_combinations_ref):
                j_indices = _find_index_in_peaks(u, v)
                j_indices_no_center = [idx for idx in j_indices if idx != center_peak_index]
                if any(idx in forbidden_indices for idx in j_indices_no_center):
                    mask[j] = False

            select_combinations_ref = select_combinations_ref[mask]
            print(select_combinations_ref)
            print(len(select_combinations_ref))

            if select_combinations_ref.size > 0:
                u_k, v_k = select_combinations_ref[0]
                final_basis_results.append([u_k, v_k])
            else:
                continue

            k_indices = _find_index_in_peaks(
                final_basis_results[1][0],
                final_basis_results[1][1],
            )
            forbidden_indices = set(k_indices)
            forbidden_indices.discard(center_peak_index)

            mask = np.ones(len(select_combinations_ref), dtype=bool)
            for j, (u, v) in enumerate(select_combinations_ref):
                j_indices = _find_index_in_peaks(u, v)
                j_indices_no_center = [idx for idx in j_indices if idx != center_peak_index]
                if any(idx in forbidden_indices for idx in j_indices_no_center):
                    mask[j] = False

            select_combinations_ref = select_combinations_ref[mask]
            print(select_combinations_ref)
            print(len(select_combinations_ref))

            H, W = im.shape
            pts_moire = generate_two_lattice_moire(
                r0=true_center,
                u1=peaks[u_1].astype(float, copy=False) - origin,
                v1=peaks[v_1].astype(float, copy=False) - origin,
                u2=peaks[u_k].astype(float, copy=False) - origin,
                v2=peaks[v_k].astype(float, copy=False) - origin,
                nrow=H,
                ncol=W,
            )

            match_r = 4.0
            matched_idx = np.array([], dtype=int)

            if pts_moire is not None and np.size(pts_moire) > 0:
                fp = np.asarray(filtered_peaks, dtype=float)
                pm = np.asarray(pts_moire, dtype=float)

                try:
                    tree = _KDTree(pm)
                    neighbors = tree.query_ball_point(fp, r=match_r)
                    matched_idx = np.flatnonzero([len(n) > 0 for n in neighbors]).astype(int)
                except Exception:
                    d2 = np.sum((fp[:, None, :] - pm[None, :, :]) ** 2, axis=2)
                    matched_idx = np.flatnonzero(np.any(d2 <= match_r**2, axis=1)).astype(int)

            print("matched_idx:", matched_idx)

            if matched_idx.size > 0 and select_combinations_ref.size > 0:
                keep_mask = ~(
                    np.isin(select_combinations_ref[:, 0], matched_idx)
                    | np.isin(select_combinations_ref[:, 1], matched_idx)
                )
                select_combinations_ref = select_combinations_ref[keep_mask]
                print(select_combinations_ref)
                print(len(select_combinations_ref))

                total_intensity_ref = []
                for u, v in select_combinations_ref:
                    unique_indices = _find_index_in_peaks(u, v)

                    if len(unique_indices) == 0:
                        total_intensity_ref.append(0.0)
                        continue

                    ui = np.unique(np.asarray(unique_indices, dtype=int))
                    d = np.linalg.norm(filtered_peaks[ui] - origin, axis=1)
                    order = np.argsort(d)
                    ui6 = ui[order[:6]]

                    total_intensity_ref.append(float(np.sum(multiple_intensity[ui6])))

                total_intensity_ref = np.asarray(total_intensity_ref)
                order_ref = np.argsort(-total_intensity_ref)
                select_combinations_ref = select_combinations_ref[order_ref]

                if select_combinations_ref.size > 0:
                    u_3, v_3 = select_combinations_ref[0]
                    final_basis_results.append([u_3, v_3])

            if len(final_basis_results) >= num:
                break

        # --- 5) Return result ---
        results = [center_peak_index]
        for basis in final_basis_results[:num]:
            results.extend([basis[0], basis[1]])

        for i in range(len(results)):
            target_peak = filtered_peaks[results[i]]
            results[i] = np.where((peaks == target_peak).all(axis=1))[0]

        print(len(results))
        print(results)
        return results


    def three_lattice_vectors(
        self,
        ind_origin=None,
        ind_u1=None,
        ind_v1=None,
        ind_u2=None,
        ind_v2=None,
        ind_u3=None,
        ind_v3=None,
        max_tile=None,
        moire_draw: bool = False,

        # Select the moire combinations to display.
        draw_moire12: bool = True,
        draw_moire13: bool = True,
        draw_moire23: bool = True,
        draw_moire123: bool = False,

        # Optionally display all three-lattice moire points; default: extra points only.
        draw_moire123_all: bool = False,

        # Default colors are numeric (r, g, b) tuples.
        color_origin=(0.0, 1.0, 0.0),
        color_lattice1=(1.0, 0.0, 0.0),
        color_lattice2=(0.0, 0.8, 1.0),
        color_lattice3=(0.0, 1.0, 0.5),

        color_moire12=(1.0, 0.5, 0.0),
        color_moire13=(0.6, 0.0, 1.0),
        color_moire23=(0.2, 0.9, 0.2),
        color_moire123_all=(0.8, 0.0, 1.0),
        color_moire123_extra=(0.0, 0.0, 0.0),

        # Opacity, marker sizes, and line widths.
        alpha_lattice_pts=0.8,
        alpha_moire_pair=0.9,
        alpha_moire123_all=0.25,
        alpha_moire123_extra=0.95,

        s_origin=90,
        s_vec_tip=45,
        s_lattice_pts=22,
        s_moire_pair=14,
        s_moire123_all=10,
        s_moire123_extra=22,

        lw_vec=1.2,

        # Retain these parameters for compatibility; inline labels are not drawn.
        vec_label_fontsize=9,
        vec_label_box_alpha=0.6,

        moire_exclude_radius_px: float = 2.0,
        moire_parent_maxlen_px: float | None = None,
        moire_round_decimals: int = 3,
        return_peaks: bool = False,
        **kwargs,
    ) -> dict[str, Any]:

        if not hasattr(self, "peaks"):
            raise RuntimeError("No peaks found. Run .find_peaks() first.")
        peaks = np.asarray(self.peaks, dtype=float)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise RuntimeError("self.peaks must have shape (N, 2) with columns [row, col].")
        if peaks.shape[0] == 0:
            raise RuntimeError("No peaks available in self.peaks.")

        im = np.asarray(self.diffraction.array)
        if im.ndim != 2:
            raise ValueError("DiffractionMoire.lattice_vectors requires a 2D diffraction array.")
        nrow, ncol = int(im.shape[0]), int(im.shape[1])

        def _as_int(x, name: str) -> int:
            if x is None:
                raise ValueError(f"{name} must be provided (int index into self.peaks).")
            xi = int(x)
            if xi < 0 or xi >= peaks.shape[0]:
                raise IndexError(f"{name}={xi} out of range for self.peaks with N={peaks.shape[0]}.")
            return xi

        def _pair_ok(a, b) -> bool:
            return (a is not None) and (b is not None)

        def _pair_half(a, b) -> bool:
            return (a is None) ^ (b is None)

        if _pair_half(ind_u1, ind_v1):
            raise ValueError("Provide both ind_u1 and ind_v1, or neither.")
        if _pair_half(ind_u2, ind_v2):
            raise ValueError("Provide both ind_u2 and ind_v2, or neither.")
        if _pair_half(ind_u3, ind_v3):
            raise ValueError("Provide both ind_u3 and ind_v3, or neither.")

        have_lattice1 = _pair_ok(ind_u1, ind_v1)
        have_lattice2 = _pair_ok(ind_u2, ind_v2)
        have_lattice3 = _pair_ok(ind_u3, ind_v3)

        if not (have_lattice1 or have_lattice2 or have_lattice3):
            raise ValueError("Provide lattice indices for at least one lattice (u/v).")

        if moire_draw and (int(have_lattice1) + int(have_lattice2) + int(have_lattice3) < 2):
            raise ValueError("moire_draw=True requires at least two lattices.")

        ind_origin_i = _as_int(ind_origin, "ind_origin")
        r0 = peaks[ind_origin_i].astype(float, copy=False)

        # --- vectors ---
        if have_lattice1:
            ind_u1_i = _as_int(ind_u1, "ind_u1")
            ind_v1_i = _as_int(ind_v1, "ind_v1")
            u1 = peaks[ind_u1_i].astype(float, copy=False) - r0
            v1 = peaks[ind_v1_i].astype(float, copy=False) - r0
        else:
            ind_u1_i = ind_v1_i = None
            u1 = v1 = None

        if have_lattice2:
            ind_u2_i = _as_int(ind_u2, "ind_u2")
            ind_v2_i = _as_int(ind_v2, "ind_v2")
            u2 = peaks[ind_u2_i].astype(float, copy=False) - r0
            v2 = peaks[ind_v2_i].astype(float, copy=False) - r0
        else:
            ind_u2_i = ind_v2_i = None
            u2 = v2 = None

        if have_lattice3:
            ind_u3_i = _as_int(ind_u3, "ind_u3")
            ind_v3_i = _as_int(ind_v3, "ind_v3")
            u3 = peaks[ind_u3_i].astype(float, copy=False) - r0
            v3 = peaks[ind_v3_i].astype(float, copy=False) - r0
        else:
            ind_u3_i = ind_v3_i = None
            u3 = v3 = None

        def _default_max_tile(u: np.ndarray, v: np.ndarray) -> int:
            du = float(np.linalg.norm(u))
            dv = float(np.linalg.norm(v))
            dmin = min(du, dv) if (du > 0 and dv > 0) else max(du, dv)
            if not np.isfinite(dmin) or dmin <= 0:
                return 10
            diag = float(np.hypot(nrow, ncol))
            mt = int(np.ceil(diag / dmin)) + 2
            return int(np.clip(mt, 3, 200))

        if max_tile is None:
            mt_list: list[int] = []
            if have_lattice1: mt_list.append(_default_max_tile(u1, v1))
            if have_lattice2: mt_list.append(_default_max_tile(u2, v2))
            if have_lattice3: mt_list.append(_default_max_tile(u3, v3))
            max_tile_use = max(mt_list) if mt_list else 10
        else:
            max_tile_use = int(max_tile)

        def _tile_points_bounded(r0: np.ndarray, u: np.ndarray, v: np.ndarray, mt: int) -> np.ndarray:
            if mt == 0:
                return r0[None, :].copy()
            a = np.arange(-mt, mt + 1, dtype=float)
            aa, bb = np.meshgrid(a, a, indexing="ij")
            pts = r0[None, None, :] + aa[..., None] * u[None, None, :] + bb[..., None] * v[None, None, :]
            pts = pts.reshape(-1, 2)
            keep = (
                (pts[:, 0] >= 0.0) & (pts[:, 0] <= (nrow - 1)) &
                (pts[:, 1] >= 0.0) & (pts[:, 1] <= (ncol - 1))
            )
            return pts[keep]

        pts1 = _tile_points_bounded(r0, u1, v1, max_tile_use) if have_lattice1 else None
        pts2 = _tile_points_bounded(r0, u2, v2, max_tile_use) if have_lattice2 else None
        pts3 = _tile_points_bounded(r0, u3, v3, max_tile_use) if have_lattice3 else None

        # === moiré sets ===
        pts_moire_12 = None
        pts_moire_13 = None
        pts_moire_23 = None
        pts_moire_123 = None
        pts_moire_123_extra = None

        if moire_draw:
            maxlen = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else float(0.5 * np.hypot(nrow, ncol))

            def _disp_set(u, v, maxlen):
                if u is None or v is None:
                    return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
                u = np.asarray(u, dtype=float).reshape(-1)
                v = np.asarray(v, dtype=float).reshape(-1)
                if u.size != 2 or v.size != 2:
                    raise ValueError(f"u,v must be 2-vectors. got u.shape={u.shape}, v.shape={v.shape}")

                dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
                if not np.isfinite(dmin) or dmin <= 0:
                    return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

                mt = int(np.clip(np.ceil(maxlen / dmin) + 1, 1, 100))
                a = np.arange(-mt, mt + 1, dtype=float)
                aa, bb = np.meshgrid(a, a, indexing="ij")
                aa, bb = aa.ravel(), bb.ravel()
                disp = aa[:, None] * u[None, :] + bb[:, None] * v[None, :]
                keep = np.linalg.norm(disp, axis=1) <= maxlen + 1e-9
                return disp[keep], np.stack([aa[keep], bb[keep]], axis=1)

            def _bounds_filter(pts):
                if pts is None or pts.size == 0:
                    return None
                keep = (
                    (pts[:, 0] >= 0.0) & (pts[:, 0] <= (nrow - 1)) &
                    (pts[:, 1] >= 0.0) & (pts[:, 1] <= (ncol - 1))
                )
                pts = pts[keep]
                return pts if pts.size else None

            def _unique_round(pts):
                if pts is None or pts.size == 0:
                    return None
                pts = np.unique(np.round(pts, int(moire_round_decimals)), axis=0)
                return pts if pts.size else None

            def _exclude_parent(pts, parent_pts, r_excl):
                if pts is None or pts.size == 0:
                    return None
                r_excl = float(r_excl)
                if r_excl <= 0:
                    return pts

                parent_pts = np.asarray(parent_pts, dtype=float)
                if parent_pts.size == 0:
                    return pts
                keep = (
                    (parent_pts[:, 0] >= 0.0) & (parent_pts[:, 0] <= (nrow - 1)) &
                    (parent_pts[:, 1] >= 0.0) & (parent_pts[:, 1] <= (ncol - 1))
                )
                parent_pts = parent_pts[keep]
                if parent_pts.size == 0:
                    return pts

                from scipy.spatial import cKDTree
                tree = cKDTree(parent_pts)
                d, _ = tree.query(pts, k=1, workers=-1)
                pts = pts[d >= r_excl]
                return pts if pts.size else None

            def _moire_two(dA, abA, dB, abB, r_excl):
                if dA.size == 0 or dB.size == 0:
                    return None
                nA, nB = int(dA.shape[0]), int(dB.shape[0])
                if nA * nB > 2_000_000:
                    raise ValueError(f"Moire two-lattice candidates too large: {nA*nB}")

                pts = r0[None, None, :] + dA[:, None, :] + dB[None, :, :]
                pts = pts.reshape(-1, 2)

                zA = (abA[:, 0] == 0) & (abA[:, 1] == 0)
                zB = (abB[:, 0] == 0) & (abB[:, 1] == 0)
                keep_combo = ~(zA[:, None] | zB[None, :]).ravel()
                pts = pts[keep_combo]

                pts = _bounds_filter(pts)
                pts = _unique_round(pts)
                if pts is None:
                    return None

                parent_pts = np.vstack((r0[None, :] + dA, r0[None, :] + dB))
                pts = _exclude_parent(pts, parent_pts, r_excl)
                return pts

            def _moire_three(d1, ab1, d2, ab2, d3, ab3, r_excl):
                if d1.size == 0 or d2.size == 0 or d3.size == 0:
                    return None
                n1, n2, n3 = int(d1.shape[0]), int(d2.shape[0]), int(d3.shape[0])
                if n1 * n2 * n3 > 2_000_000:
                    raise ValueError(f"Moire three-lattice candidates too large: {n1*n2*n3}")

                pts = r0[None, None, None, :] + d1[:, None, None, :] + d2[None, :, None, :] + d3[None, None, :, :]
                pts = pts.reshape(-1, 2)

                z1 = (ab1[:, 0] == 0) & (ab1[:, 1] == 0)
                z2 = (ab2[:, 0] == 0) & (ab2[:, 1] == 0)
                z3 = (ab3[:, 0] == 0) & (ab3[:, 1] == 0)
                is_parent = (
                    (z1[:, None, None] & z2[None, :, None]) |
                    (z1[:, None, None] & z3[None, None, :]) |
                    (z2[None, :, None] & z3[None, None, :])
                ).ravel()
                pts = pts[~is_parent]

                pts = _bounds_filter(pts)
                pts = _unique_round(pts)
                if pts is None:
                    return None

                parent_pts = np.vstack((r0[None, :] + d1, r0[None, :] + d2, r0[None, :] + d3))
                pts = _exclude_parent(pts, parent_pts, r_excl)
                return pts

            def _pts_key(pts):
                pr = np.ascontiguousarray(np.round(pts, int(moire_round_decimals)))
                return pr.view(np.dtype((np.void, pr.dtype.itemsize * pr.shape[1]))).ravel()

            need_d1 = have_lattice1 and (draw_moire12 or draw_moire13 or draw_moire123)
            need_d2 = have_lattice2 and (draw_moire12 or draw_moire23 or draw_moire123)
            need_d3 = have_lattice3 and (draw_moire13 or draw_moire23 or draw_moire123)

            d1, ab1 = _disp_set(u1, v1, maxlen) if need_d1 else (np.zeros((0, 2)), np.zeros((0, 2)))
            d2, ab2 = _disp_set(u2, v2, maxlen) if need_d2 else (np.zeros((0, 2)), np.zeros((0, 2)))
            d3, ab3 = _disp_set(u3, v3, maxlen) if need_d3 else (np.zeros((0, 2)), np.zeros((0, 2)))

            if draw_moire12 and have_lattice1 and have_lattice2:
                pts_moire_12 = _moire_two(d1, ab1, d2, ab2, moire_exclude_radius_px)
            if draw_moire13 and have_lattice1 and have_lattice3:
                pts_moire_13 = _moire_two(d1, ab1, d3, ab3, moire_exclude_radius_px)
            if draw_moire23 and have_lattice2 and have_lattice3:
                pts_moire_23 = _moire_two(d2, ab2, d3, ab3, moire_exclude_radius_px)

            if draw_moire123 and have_lattice1 and have_lattice2 and have_lattice3:
                pts_moire_123 = _moire_three(d1, ab1, d2, ab2, d3, ab3, moire_exclude_radius_px)

                pair_list = [p for p in (pts_moire_12, pts_moire_13, pts_moire_23) if p is not None and p.size]
                if pts_moire_123 is not None and pts_moire_123.size:
                    if pair_list:
                        pair_all = np.unique(np.vstack(pair_list), axis=0)
                        k_pair = _pts_key(pair_all)
                        k_123 = _pts_key(pts_moire_123)
                        extra_mask = ~np.isin(k_123, k_pair)
                        pts_moire_123_extra = pts_moire_123[extra_mask] if np.any(extra_mask) else None
                    else:
                        pts_moire_123_extra = pts_moire_123

        data: dict[str, Any] = {
            "origin_index": ind_origin_i,
            "origin_rc": r0.copy(),
            "max_tile": max_tile_use,
            "lattice1_points_rc": pts1,
            "lattice2_points_rc": pts2,
            "lattice3_points_rc": pts3,
            "moire12_points_rc": pts_moire_12,
            "moire13_points_rc": pts_moire_13,
            "moire23_points_rc": pts_moire_23,
            "moire123_points_rc": pts_moire_123,
            "moire123_extra_points_rc": pts_moire_123_extra,
            "moire_points_rc": pts_moire_123,
        }

        if have_lattice1:
            data.update({"u1_index": ind_u1_i, "v1_index": ind_v1_i, "u1_rc": u1, "v1_rc": v1})
        if have_lattice2:
            data.update({"u2_index": ind_u2_i, "v2_index": ind_v2_i, "u2_rc": u2, "v2_rc": v2})
        if have_lattice3:
            data.update({"u3_index": ind_u3_i, "v3_index": ind_v3_i, "u3_rc": u3, "v3_rc": v3})

        self.lattice_vectors_data = data

        fig, ax = show_2d(im, **kwargs)
        ax0 = ax.flat[0] if hasattr(ax, "flat") else ax

        ax0.scatter([r0[1]], [r0[0]], s=s_origin, marker="x", c=[color_origin], linewidths=lw_vec, zorder=10)

        # Draw basis vectors without inline u/v labels.
        def _draw_vec(vec, color):
            p = r0 + vec
            ax0.plot([r0[1], p[1]], [r0[0], p[0]], color=color, linewidth=lw_vec, zorder=9)
            ax0.scatter([p[1]], [p[0]], s=s_vec_tip, marker="o",
                        facecolors="none", edgecolors=[color], linewidths=lw_vec, zorder=10)

        if have_lattice1 and pts1 is not None and pts1.size:
            _draw_vec(u1, color_lattice1); _draw_vec(v1, color_lattice1)
            ax0.scatter(pts1[:, 1], pts1[:, 0], s=s_lattice_pts, marker="o",
                        facecolors="none", edgecolors=[color_lattice1],
                        alpha=alpha_lattice_pts, zorder=5, label="lattice1")

        if have_lattice2 and pts2 is not None and pts2.size:
            _draw_vec(u2, color_lattice2); _draw_vec(v2, color_lattice2)
            ax0.scatter(pts2[:, 1], pts2[:, 0], s=s_lattice_pts, marker="o",
                        facecolors="none", edgecolors=[color_lattice2],
                        alpha=alpha_lattice_pts, zorder=5, label="lattice2")

        if have_lattice3 and pts3 is not None and pts3.size:
            _draw_vec(u3, color_lattice3); _draw_vec(v3, color_lattice3)
            ax0.scatter(pts3[:, 1], pts3[:, 0], s=s_lattice_pts, marker="o",
                        facecolors="none", edgecolors=[color_lattice3],
                        alpha=alpha_lattice_pts, zorder=5, label="lattice3")

        n12 = n13 = n23 = n123 = nX = 0

        if pts_moire_12 is not None and pts_moire_12.size:
            ax0.scatter(pts_moire_12[:, 1], pts_moire_12[:, 0], s=s_moire_pair, marker="o",
                        facecolors="none", edgecolors=[color_moire12],
                        alpha=alpha_moire_pair, zorder=6, label="moire(1-2)")
            n12 = pts_moire_12.shape[0]

        if pts_moire_13 is not None and pts_moire_13.size:
            ax0.scatter(pts_moire_13[:, 1], pts_moire_13[:, 0], s=s_moire_pair, marker="o",
                        facecolors="none", edgecolors=[color_moire13],
                        alpha=alpha_moire_pair, zorder=6, label="moire(1-3)")
            n13 = pts_moire_13.shape[0]

        if pts_moire_23 is not None and pts_moire_23.size:
            ax0.scatter(pts_moire_23[:, 1], pts_moire_23[:, 0], s=s_moire_pair, marker="o",
                        facecolors="none", edgecolors=[color_moire23],
                        alpha=alpha_moire_pair, zorder=6, label="moire(2-3)")
            n23 = pts_moire_23.shape[0]

        if (pts_moire_123 is not None) and pts_moire_123.size:
            n123 = pts_moire_123.shape[0]
            if draw_moire123_all:
                ax0.scatter(pts_moire_123[:, 1], pts_moire_123[:, 0], s=s_moire123_all, marker=".",
                            c=[color_moire123_all], alpha=alpha_moire123_all, zorder=4,
                            label="moire(1-2-3) all")

        if pts_moire_123_extra is not None and pts_moire_123_extra.size:
            ax0.scatter(pts_moire_123_extra[:, 1], pts_moire_123_extra[:, 0], s=s_moire123_extra, marker="x",
                        c=[color_moire123_extra], alpha=alpha_moire123_extra, zorder=7,
                        label="moire(1-2-3) extra")
            nX = pts_moire_123_extra.shape[0]

        handles, labels = ax0.get_legend_handles_labels()
        if labels:
            ax0.legend(loc="upper right", fontsize=9, frameon=True)

        info = f"moire12={n12}, moire13={n13}, moire23={n23}, moire123={n123}, extra={nX}"
        ax0.text(
            0.01, 0.99, info,
            transform=ax0.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            zorder=20,
        )

        ax0.figure.canvas.draw_idle()
        return {"data": data, "fig": fig, "ax": ax0} if return_peaks else {}

# Initialization and templates

def _as_scalar_int(x):
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like index, got shape={arr.shape}")
    return int(arr.ravel()[0])

def sanitize_vector_indices(vector_indices):
    return [_as_scalar_int(x) for x in vector_indices]

def get_auto_init_from_moire_three(moire, vector_indices):
    peaks = np.asarray(moire.peaks, dtype=float)
    vector_indices = sanitize_vector_indices(vector_indices)
    if len(vector_indices) != 7:
        raise ValueError(f"Expected 7 indices [origin,u1,v1,u2,v2,u3,v3], got {len(vector_indices)}")

    ind_origin, ind_u1, ind_v1, ind_u2, ind_v2, ind_u3, ind_v3 = vector_indices

    origin_rc = peaks[ind_origin].copy()
    u1_rc = peaks[ind_u1].copy() - origin_rc
    v1_rc = peaks[ind_v1].copy() - origin_rc
    u2_rc = peaks[ind_u2].copy() - origin_rc
    v2_rc = peaks[ind_v2].copy() - origin_rc
    u3_rc = peaks[ind_u3].copy() - origin_rc
    v3_rc = peaks[ind_v3].copy() - origin_rc

    return {
        "origin_rc": origin_rc,
        "u1_rc": u1_rc,
        "v1_rc": v1_rc,
        "u2_rc": u2_rc,
        "v2_rc": v2_rc,
        "u3_rc": u3_rc,
        "v3_rc": v3_rc,
        "seed_peak_indices": [ind_origin, ind_u1, ind_v1, ind_u2, ind_v2, ind_u3, ind_v3],
    }

def make_template_from_peaks(image, peak_list_rc, radius_px, subtract_quantile=30.0):
    peak_list_rc = np.asarray(peak_list_rc, dtype=float)
    if peak_list_rc.ndim == 1:
        peak_list_rc = peak_list_rc[None, :]
    if len(peak_list_rc) == 0:
        raise ValueError("peak_list_rc is empty")

    patches = []
    for peak_rc in peak_list_rc:
        p = extract_patch(image, peak_rc, radius_px)
        bg = np.percentile(p, subtract_quantile)
        p = np.clip(p - bg, 0, None)
        if np.max(p) > 0:
            p = p / np.max(p)
        patches.append(p)

    template = np.mean(np.stack(patches, axis=0), axis=0)
    if np.max(template) > 0:
        template = template / np.max(template)
    return template.astype(np.float32)

# Three-lattice geometry

def _unique_points_or_none(*arrays, decimals=3):
    valid = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        valid.append(arr.reshape(-1, 2))
    if not valid:
        return None
    pts = np.vstack(valid)
    pts = np.unique(np.round(pts, int(decimals)), axis=0)
    return pts if pts.size else None

def build_three_lattice_moire_from_vectors(
    origin_rc,
    u1_rc,
    v1_rc,
    u2_rc,
    v2_rc,
    u3_rc,
    v3_rc,
    shape,
    *,
    max_tile=None,
    draw_moire12=True,
    draw_moire13=True,
    draw_moire23=False,
    draw_moire123=False,
    draw_moire123_extra=False,
    moire_exclude_radius_px=2.0,
    moire_parent_maxlen_px=240.0,
    moire_round_decimals=3,
):
    """
    No-plot geometry helper for fitting.

    This mirrors the geometric point generation logic used in
    DiffractionMoire.three_lattice_vectors(...), but skips plotting so it can be
    called inside optimization loops.
    """
    origin_rc = np.asarray(origin_rc, dtype=float).reshape(2)
    u1_rc = np.asarray(u1_rc, dtype=float).reshape(2)
    v1_rc = np.asarray(v1_rc, dtype=float).reshape(2)
    u2_rc = np.asarray(u2_rc, dtype=float).reshape(2)
    v2_rc = np.asarray(v2_rc, dtype=float).reshape(2)
    u3_rc = np.asarray(u3_rc, dtype=float).reshape(2)
    v3_rc = np.asarray(v3_rc, dtype=float).reshape(2)

    nrow, ncol = int(shape[0]), int(shape[1])

    def _default_max_tile(u, v):
        du = float(np.linalg.norm(u))
        dv = float(np.linalg.norm(v))
        dmin = min(du, dv) if (du > 0 and dv > 0) else max(du, dv)
        if not np.isfinite(dmin) or dmin <= 0:
            return 10
        diag = float(np.hypot(nrow, ncol))
        mt = int(np.ceil(diag / dmin)) + 2
        return int(np.clip(mt, 3, 200))

    if max_tile is None:
        max_tile_use = max(
            _default_max_tile(u1_rc, v1_rc),
            _default_max_tile(u2_rc, v2_rc),
            _default_max_tile(u3_rc, v3_rc),
        )
    else:
        max_tile_use = int(max_tile)

    def _tile_points_bounded(r0, u, v, mt):
        if mt == 0:
            return r0[None, :].copy()
        a = np.arange(-mt, mt + 1, dtype=float)
        aa, bb = np.meshgrid(a, a, indexing="ij")
        pts = r0[None, None, :] + aa[..., None] * u[None, None, :] + bb[..., None] * v[None, None, :]
        pts = pts.reshape(-1, 2)
        keep = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] <= (nrow - 1)) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] <= (ncol - 1))
        )
        return pts[keep]

    pts1 = _tile_points_bounded(origin_rc, u1_rc, v1_rc, max_tile_use)
    pts2 = _tile_points_bounded(origin_rc, u2_rc, v2_rc, max_tile_use)
    pts3 = _tile_points_bounded(origin_rc, u3_rc, v3_rc, max_tile_use)

    maxlen = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else float(0.5 * np.hypot(nrow, ncol))

    def _disp_set(u, v, maxlen):
        dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
        if not np.isfinite(dmin) or dmin <= 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

        mt = int(np.clip(np.ceil(maxlen / dmin) + 1, 1, 100))
        a = np.arange(-mt, mt + 1, dtype=float)
        aa, bb = np.meshgrid(a, a, indexing="ij")
        aa, bb = aa.ravel(), bb.ravel()
        disp = aa[:, None] * u[None, :] + bb[:, None] * v[None, :]
        keep = np.linalg.norm(disp, axis=1) <= maxlen + 1e-9
        return disp[keep], np.stack([aa[keep], bb[keep]], axis=1)

    def _bounds_filter(pts):
        if pts is None or pts.size == 0:
            return None
        keep = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] <= (nrow - 1)) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] <= (ncol - 1))
        )
        pts = pts[keep]
        return pts if pts.size else None

    def _unique_round(pts):
        if pts is None or pts.size == 0:
            return None
        pts = np.unique(np.round(pts, int(moire_round_decimals)), axis=0)
        return pts if pts.size else None

    def _exclude_parent(pts, parent_pts, r_excl):
        if pts is None or pts.size == 0:
            return None
        r_excl = float(r_excl)
        if r_excl <= 0:
            return pts

        parent_pts = np.asarray(parent_pts, dtype=float)
        if parent_pts.size == 0:
            return pts

        keep = (
            (parent_pts[:, 0] >= 0.0) & (parent_pts[:, 0] <= (nrow - 1)) &
            (parent_pts[:, 1] >= 0.0) & (parent_pts[:, 1] <= (ncol - 1))
        )
        parent_pts = parent_pts[keep]
        if parent_pts.size == 0:
            return pts

        from scipy.spatial import cKDTree
        tree = cKDTree(parent_pts)
        d, _ = tree.query(pts, k=1)
        pts = pts[d >= r_excl]
        return pts if pts.size else None

    def _moire_two(dA, abA, dB, abB, r_excl):
        if dA.size == 0 or dB.size == 0:
            return None
        nA, nB = int(dA.shape[0]), int(dB.shape[0])
        if nA * nB > 2_000_000:
            raise ValueError(f"Moire two-lattice candidates too large: {nA*nB}")

        pts = origin_rc[None, None, :] + dA[:, None, :] + dB[None, :, :]
        pts = pts.reshape(-1, 2)

        zA = (abA[:, 0] == 0) & (abA[:, 1] == 0)
        zB = (abB[:, 0] == 0) & (abB[:, 1] == 0)
        keep_combo = ~(zA[:, None] | zB[None, :]).ravel()
        pts = pts[keep_combo]

        pts = _bounds_filter(pts)
        pts = _unique_round(pts)
        if pts is None:
            return None

        parent_pts = np.vstack((origin_rc[None, :] + dA, origin_rc[None, :] + dB))
        pts = _exclude_parent(pts, parent_pts, r_excl)
        return pts

    def _moire_three(d1, ab1, d2, ab2, d3, ab3, r_excl):
        if d1.size == 0 or d2.size == 0 or d3.size == 0:
            return None
        n1, n2, n3 = int(d1.shape[0]), int(d2.shape[0]), int(d3.shape[0])
        if n1 * n2 * n3 > 2_000_000:
            raise ValueError(f"Moire three-lattice candidates too large: {n1*n2*n3}")

        pts = origin_rc[None, None, None, :] + d1[:, None, None, :] + d2[None, :, None, :] + d3[None, None, :, :]
        pts = pts.reshape(-1, 2)

        z1 = (ab1[:, 0] == 0) & (ab1[:, 1] == 0)
        z2 = (ab2[:, 0] == 0) & (ab2[:, 1] == 0)
        z3 = (ab3[:, 0] == 0) & (ab3[:, 1] == 0)
        is_parent = (
            (z1[:, None, None] & z2[None, :, None]) |
            (z1[:, None, None] & z3[None, None, :]) |
            (z2[None, :, None] & z3[None, None, :])
        ).ravel()
        pts = pts[~is_parent]

        pts = _bounds_filter(pts)
        pts = _unique_round(pts)
        if pts is None:
            return None

        parent_pts = np.vstack((origin_rc[None, :] + d1, origin_rc[None, :] + d2, origin_rc[None, :] + d3))
        pts = _exclude_parent(pts, parent_pts, r_excl)
        return pts

    def _pts_key(pts):
        pr = np.ascontiguousarray(np.round(pts, int(moire_round_decimals)))
        return pr.view(np.dtype((np.void, pr.dtype.itemsize * pr.shape[1]))).ravel()

    need_d1 = bool(draw_moire12 or draw_moire13 or draw_moire123 or draw_moire123_extra)
    need_d2 = bool(draw_moire12 or draw_moire23 or draw_moire123 or draw_moire123_extra)
    need_d3 = bool(draw_moire13 or draw_moire23 or draw_moire123 or draw_moire123_extra)

    d1, ab1 = _disp_set(u1_rc, v1_rc, maxlen) if need_d1 else (np.zeros((0, 2)), np.zeros((0, 2)))
    d2, ab2 = _disp_set(u2_rc, v2_rc, maxlen) if need_d2 else (np.zeros((0, 2)), np.zeros((0, 2)))
    d3, ab3 = _disp_set(u3_rc, v3_rc, maxlen) if need_d3 else (np.zeros((0, 2)), np.zeros((0, 2)))

    pts_moire_12 = _moire_two(d1, ab1, d2, ab2, moire_exclude_radius_px) if draw_moire12 else None
    pts_moire_13 = _moire_two(d1, ab1, d3, ab3, moire_exclude_radius_px) if draw_moire13 else None
    pts_moire_23 = _moire_two(d2, ab2, d3, ab3, moire_exclude_radius_px) if draw_moire23 else None

    pts_moire_123 = None
    pts_moire_123_extra = None
    if draw_moire123 or draw_moire123_extra:
        pts_moire_123 = _moire_three(d1, ab1, d2, ab2, d3, ab3, moire_exclude_radius_px)
        pair_list = [p for p in (pts_moire_12, pts_moire_13, pts_moire_23) if p is not None and p.size]
        if pts_moire_123 is not None and pts_moire_123.size:
            if pair_list:
                pair_all = np.unique(np.vstack(pair_list), axis=0)
                k_pair = _pts_key(pair_all)
                k_123 = _pts_key(pts_moire_123)
                extra_mask = ~np.isin(k_123, k_pair)
                pts_moire_123_extra = pts_moire_123[extra_mask] if np.any(extra_mask) else None
            else:
                pts_moire_123_extra = pts_moire_123

    active_parts = []
    if draw_moire12 and pts_moire_12 is not None:
        active_parts.append(pts_moire_12)
    if draw_moire13 and pts_moire_13 is not None:
        active_parts.append(pts_moire_13)
    if draw_moire23 and pts_moire_23 is not None:
        active_parts.append(pts_moire_23)
    if draw_moire123 and pts_moire_123 is not None:
        active_parts.append(pts_moire_123)
    if draw_moire123_extra and pts_moire_123_extra is not None:
        active_parts.append(pts_moire_123_extra)

    moire_active_points_rc = _unique_points_or_none(*active_parts, decimals=moire_round_decimals)

    return {
        "origin_rc": origin_rc.copy(),
        "max_tile": max_tile_use,
        "lattice1_points_rc": pts1,
        "lattice2_points_rc": pts2,
        "lattice3_points_rc": pts3,
        "moire12_points_rc": pts_moire_12,
        "moire13_points_rc": pts_moire_13,
        "moire23_points_rc": pts_moire_23,
        "moire123_points_rc": pts_moire_123,
        "moire123_extra_points_rc": pts_moire_123_extra,
        "moire_points_rc": pts_moire_123,
        "moire_active_points_rc": moire_active_points_rc,
        "moire_info": {
            "moire_parent_maxlen_px": maxlen,
            "moire_exclude_radius_px": float(moire_exclude_radius_px),
            "moire_round_decimals": int(moire_round_decimals),
            "active_components": {
                "draw_moire12": bool(draw_moire12),
                "draw_moire13": bool(draw_moire13),
                "draw_moire23": bool(draw_moire23),
                "draw_moire123": bool(draw_moire123),
                "draw_moire123_extra": bool(draw_moire123_extra),
            },
            "n_moire12": 0 if pts_moire_12 is None else int(len(pts_moire_12)),
            "n_moire13": 0 if pts_moire_13 is None else int(len(pts_moire_13)),
            "n_moire23": 0 if pts_moire_23 is None else int(len(pts_moire_23)),
            "n_moire123": 0 if pts_moire_123 is None else int(len(pts_moire_123)),
            "n_moire123_extra": 0 if pts_moire_123_extra is None else int(len(pts_moire_123_extra)),
            "n_moire_active": 0 if moire_active_points_rc is None else int(len(moire_active_points_rc)),
        },
    }

# Three-lattice fitter

class TripleLatticeMoireFitter:
    """
    Three parent lattices are trained directly.

    Explicit moiré points are generated from the same three-lattice geometry
    logic as your later `three_lattice_vectors(...)` workflow, then a weak
    moiré template is rendered at those positions during the final fitting stage.
    """

    def __init__(self, image, device="cpu", dtype=torch.float32, loss_mode="log1p"):
        self.image_raw = np.asarray(image, dtype=np.float32)
        self.image = robust_normalize(self.image_raw)
        self.shape = self.image.shape

        self.device = torch.device(device)
        self.dtype = dtype
        self.loss_mode = loss_mode

        self.target_t = torch.as_tensor(self.image, device=self.device, dtype=self.dtype)
        self.mask_t = None

        self.origin = None
        self.model = None
        self.ctx = None

        self.center_disk = None
        self.lat1 = None
        self.lat2 = None
        self.lat3 = None
        self.dc_bg = None
        self.gaussian_bg = None

        self.moire_template_raw = None
        self.moire_scale_raw = None
        self.use_moire_component = False

        self._origin_init = None
        self._origin_delta = None

        self.moire_flags = {
            "draw_moire12": True,
            "draw_moire13": True,
            "draw_moire23": False,
            "draw_moire123": False,
            "draw_moire123_extra": False,
        }
        self.moire_max_tile = None
        self.moire_exclude_radius_px = 2.0
        self.moire_parent_maxlen_px = 240.0
        self.moire_round_decimals = 3
        self.moire_tv_weight = 1e-4
        self.refine_moire_template = True

        self.loss_history = []

    def set_mask(self, mask):
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} != image shape {self.shape}")
        self.mask_t = torch.as_tensor(mask, device=self.device, dtype=self.dtype)

    def _transform(self, x):
        if self.loss_mode == "linear":
            return x
        if self.loss_mode == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0) + 1e-8)
        if self.loss_mode == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        raise ValueError(f"unknown loss_mode: {self.loss_mode}")

    def _masked_mse(self, a, b):
        err2 = (a - b) ** 2
        if self.mask_t is None:
            return err2.mean()
        w = self.mask_t
        return (err2 * w).sum() / torch.clamp(w.sum(), min=1e-8)

    def _tv_loss(self, t):
        if t is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        dx = torch.abs(t[1:, :] - t[:-1, :]).mean()
        dy = torch.abs(t[:, 1:] - t[:, :-1]).mean()
        return dx + dy

    def _current_parent_summary_numpy(self):
        return {
            "origin_rc": self.origin.coords.detach().cpu().numpy().copy(),
            "u1_rc": np.array([float(self.lat1.u_row.detach().cpu()), float(self.lat1.u_col.detach().cpu())]),
            "v1_rc": np.array([float(self.lat1.v_row.detach().cpu()), float(self.lat1.v_col.detach().cpu())]),
            "u2_rc": np.array([float(self.lat2.u_row.detach().cpu()), float(self.lat2.u_col.detach().cpu())]),
            "v2_rc": np.array([float(self.lat2.v_row.detach().cpu()), float(self.lat2.v_col.detach().cpu())]),
            "u3_rc": np.array([float(self.lat3.u_row.detach().cpu()), float(self.lat3.u_col.detach().cpu())]),
            "v3_rc": np.array([float(self.lat3.v_row.detach().cpu()), float(self.lat3.v_col.detach().cpu())]),
        }

    def get_trained_moire_data(self):
        if any(x is None for x in (self.origin, self.lat1, self.lat2, self.lat3)):
            return None
        s = self._current_parent_summary_numpy()
        return build_three_lattice_moire_from_vectors(
            origin_rc=s["origin_rc"],
            u1_rc=s["u1_rc"],
            v1_rc=s["v1_rc"],
            u2_rc=s["u2_rc"],
            v2_rc=s["v2_rc"],
            u3_rc=s["u3_rc"],
            v3_rc=s["v3_rc"],
            shape=self.shape,
            max_tile=self.moire_max_tile,
            draw_moire12=self.moire_flags["draw_moire12"],
            draw_moire13=self.moire_flags["draw_moire13"],
            draw_moire23=self.moire_flags["draw_moire23"],
            draw_moire123=self.moire_flags["draw_moire123"],
            draw_moire123_extra=self.moire_flags["draw_moire123_extra"],
            moire_exclude_radius_px=self.moire_exclude_radius_px,
            moire_parent_maxlen_px=self.moire_parent_maxlen_px,
            moire_round_decimals=self.moire_round_decimals,
        )

    def get_trained_moire_points(self):
        data = self.get_trained_moire_data()
        if data is None:
            return None
        return data.get("moire_active_points_rc", None)

    def _render_template_at_points(self, template_t, points_rc, scale_t):
        canvas = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        if points_rc is None or len(points_rc) == 0:
            return canvas

        tpl = torch.clamp(template_t, min=0.0)
        tpl_max = torch.clamp(tpl.max(), min=1e-8)
        tpl = tpl / tpl_max
        scale = torch.clamp(scale_t, min=0.0)

        th, tw = int(tpl.shape[0]), int(tpl.shape[1])
        cr, cc = th // 2, tw // 2

        for pr, pc in np.asarray(points_rc, dtype=float):
            r = int(np.round(pr))
            c = int(np.round(pc))

            r0 = max(0, r - cr)
            c0 = max(0, c - cc)
            # Clip the original footprint; clipping its start must not shift its end.
            r1 = min(self.shape[0], r - cr + th)
            c1 = min(self.shape[1], c - cc + tw)

            tr0 = max(0, cr - r)
            tc0 = max(0, cc - c)
            tr1 = tr0 + (r1 - r0)
            tc1 = tc0 + (c1 - c0)

            if r1 > r0 and c1 > c0 and tr1 > tr0 and tc1 > tc0:
                canvas[r0:r1, c0:c1] = canvas[r0:r1, c0:c1] + scale * tpl[tr0:tr1, tc0:tc1]

        return canvas

    def render_moire_torch(self):
        if (self.moire_template_raw is None) or (self.moire_scale_raw is None) or (not self.use_moire_component):
            return torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        points_rc = self.get_trained_moire_points()
        return self._render_template_at_points(self.moire_template_raw, points_rc, self.moire_scale_raw)

    def _make_spot_disk(self, name, template, refine_spot_templates, tv_weight):
        disk = DiskTemplate.from_array(
            name=name,
            array=template,
            refine_all_pixels=bool(refine_spot_templates),
            normalize="none",
            intensity=0.0,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        disk.set_origin(self.origin)
        return disk

    def _make_lattice(self, name, disk, u_rc, v_rc, uv_delta_px, lattice_order, spot_intensity, boundary_px):
        lat = SyntheticDiskLattice(
            name=name,
            disk=disk,
            u_row=(float(u_rc[0]), float(uv_delta_px)),
            u_col=(float(u_rc[1]), float(uv_delta_px)),
            v_row=(float(v_rc[0]), float(uv_delta_px)),
            v_col=(float(v_rc[1]), float(uv_delta_px)),
            u_max=int(lattice_order),
            v_max=int(lattice_order),
            per_disk_intensity=True,
            per_disk_slopes=False,
            intensity_0=spot_intensity,
            center_intensity_0=0.0,
            exclude_indices=None,
            boundary_px=float(boundary_px),
        )
        lat.set_origin(self.origin)
        return lat

    def build_from_auto_init(
        self,
        origin_rc,
        u1_rc,
        v1_rc,
        u2_rc,
        v2_rc,
        u3_rc,
        v3_rc,
        peak_seed_rc,
        center_radius_px=9,
        spot_radius_px=5,
        origin_delta_px=6.0,
        uv_delta_px=4.0,
        lattice_order=6,
        center_intensity=1.0,
        spot_intensity=0.12,
        moire_intensity=0.02,
        max_tile=None,
        draw_moire12=True,
        draw_moire13=True,
        draw_moire23=False,
        draw_moire123=False,
        draw_moire123_extra=False,
        moire_exclude_radius_px=2.0,
        moire_parent_maxlen_px=240.0,
        moire_round_decimals=3,
        gaussian_bg_sigma=70.0,
        gaussian_bg_intensity=0.03,
        dc_bg_intensity=0.005,
        refine_center_template=False,
        refine_spot_templates=True,
        refine_moire_template=True,
        tv_weight=1e-4,
        boundary_px=7.0,
    ):
        origin_rc = np.asarray(origin_rc, dtype=float)
        u1_rc = np.asarray(u1_rc, dtype=float)
        v1_rc = np.asarray(v1_rc, dtype=float)
        u2_rc = np.asarray(u2_rc, dtype=float)
        v2_rc = np.asarray(v2_rc, dtype=float)
        u3_rc = np.asarray(u3_rc, dtype=float)
        v3_rc = np.asarray(v3_rc, dtype=float)
        peak_seed_rc = np.asarray(peak_seed_rc, dtype=float)

        self._origin_init = origin_rc.copy()
        self._origin_delta = float(origin_delta_px)

        self.moire_flags = {
            "draw_moire12": bool(draw_moire12),
            "draw_moire13": bool(draw_moire13),
            "draw_moire23": bool(draw_moire23),
            "draw_moire123": bool(draw_moire123),
            "draw_moire123_extra": bool(draw_moire123_extra),
        }
        self.moire_max_tile = max_tile
        self.moire_exclude_radius_px = float(moire_exclude_radius_px)
        self.moire_parent_maxlen_px = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else None
        self.moire_round_decimals = int(moire_round_decimals)
        self.moire_tv_weight = float(tv_weight)
        self.refine_moire_template = bool(refine_moire_template)

        center_template = make_template_from_peaks(
            self.image_raw,
            [origin_rc],
            radius_px=center_radius_px,
            subtract_quantile=20.0,
        )

        spot_template = make_template_from_peaks(
            self.image_raw,
            peak_seed_rc,
            radius_px=spot_radius_px,
            subtract_quantile=30.0,
        )

        moire_template = spot_template.copy()

        self.origin = OriginND(ndim=2, init=[float(origin_rc[0]), float(origin_rc[1])]).to(self.device)

        self.center_disk = DiskTemplate.from_array(
            name="center_disk",
            array=center_template,
            refine_all_pixels=bool(refine_center_template),
            normalize="none",
            intensity=center_intensity,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        self.center_disk.set_origin(self.origin)

        disk1 = self._make_spot_disk("spot_disk1", spot_template, refine_spot_templates, tv_weight)
        disk2 = self._make_spot_disk("spot_disk2", spot_template, refine_spot_templates, tv_weight)
        disk3 = self._make_spot_disk("spot_disk3", spot_template, refine_spot_templates, tv_weight)

        self.lat1 = self._make_lattice("lat1", disk1, u1_rc, v1_rc, uv_delta_px, lattice_order, spot_intensity, boundary_px)
        self.lat2 = self._make_lattice("lat2", disk2, u2_rc, v2_rc, uv_delta_px, lattice_order, spot_intensity, boundary_px)
        self.lat3 = self._make_lattice("lat3", disk3, u3_rc, v3_rc, uv_delta_px, lattice_order, spot_intensity, boundary_px)

        self.dc_bg = DCBackground(name="dc_background", intensity=dc_bg_intensity)
        self.gaussian_bg = GaussianBackground(name="gaussian_background", sigma=gaussian_bg_sigma, intensity=gaussian_bg_intensity)
        self.gaussian_bg.set_origin(self.origin)

        components = [
            self.center_disk,
            self.dc_bg,
            self.gaussian_bg,
            self.lat1,
            self.lat2,
            self.lat3,
        ]
        self.model = AdditiveRenderModel(origin=self.origin, components=components).to(self.device)
        self.ctx = RenderContext(shape=self.shape, device=self.device, dtype=self.dtype, mask=self.mask_t)

        self.moire_template_raw = torch.nn.Parameter(torch.as_tensor(moire_template, device=self.device, dtype=self.dtype))
        self.moire_scale_raw = torch.nn.Parameter(torch.tensor(float(moire_intensity), device=self.device, dtype=self.dtype))
        self.moire_template_raw.requires_grad_(bool(refine_moire_template))
        self.moire_scale_raw.requires_grad_(True)

        self.use_moire_component = False

    def render_torch(self):
        if self.model is None:
            raise RuntimeError("Call build_from_auto_init() first.")
        pred = self.model(self.ctx)
        if self.use_moire_component:
            pred = pred + self.render_moire_torch()
        return pred

    def render_numpy(self):
        with torch.no_grad():
            return self.render_torch().detach().cpu().numpy()

    def _enforce_origin_bounds(self):
        if self._origin_init is None or self._origin_delta is None:
            return
        with torch.no_grad():
            self.origin.coords[0].clamp_(float(self._origin_init[0] - self._origin_delta), float(self._origin_init[0] + self._origin_delta))
            self.origin.coords[1].clamp_(float(self._origin_init[1] - self._origin_delta), float(self._origin_init[1] + self._origin_delta))

    def fit(self, n_steps=150, lr=5e-2, constraint_weight=1.0, print_every=25):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.moire_template_raw is not None and self.moire_template_raw.requires_grad:
            params.append(self.moire_template_raw)
        if self.moire_scale_raw is not None and self.moire_scale_raw.requires_grad:
            params.append(self.moire_scale_raw)

        optimizer = torch.optim.Adam(params, lr=lr)
        target_t = self._transform(self.target_t)
        hist = []

        for step in range(n_steps):
            optimizer.zero_grad()

            pred = self.render_torch()
            pred_t = self._transform(pred)

            data_loss = self._masked_mse(pred_t, target_t)
            reg_loss = self.model.total_constraint_loss(self.ctx)

            if self.moire_template_raw is not None:
                reg_loss = reg_loss + self.moire_tv_weight * self._tv_loss(torch.clamp(self.moire_template_raw, min=0.0))

            loss = data_loss + float(constraint_weight) * reg_loss
            loss.backward()
            optimizer.step()
            self.model.apply_hard_constraints(self.ctx)
            self._enforce_origin_bounds()

            with torch.no_grad():
                if self.moire_template_raw is not None:
                    self.moire_template_raw.clamp_(min=0.0)
                if self.moire_scale_raw is not None:
                    self.moire_scale_raw.clamp_(min=0.0)

            loss_val = float(loss.detach().cpu())
            hist.append(loss_val)

            if print_every and ((step == 0) or ((step + 1) % print_every == 0) or (step == n_steps - 1)):
                msg = (
                    f"step {step+1:4d}/{n_steps} | "
                    f"loss={loss_val:.6e} | "
                    f"data={float(data_loss.detach().cpu()):.6e} | "
                    f"reg={float(reg_loss.detach().cpu()):.6e}"
                )
                if self.use_moire_component and self.moire_scale_raw is not None:
                    msg += f" | moire_scale={float(self.moire_scale_raw.detach().cpu()):.4f}"
                print(msg)

        self.loss_history.extend(hist)
        return hist

    def _set_requires_grad(self, tensor, enabled):
        tensor.requires_grad_(bool(enabled))

    def fit_staged(
        self,
        stage1_steps=40,
        stage2_steps=120,
        stage3_steps=120,
        lr1=2e-1,
        lr2=5e-2,
        lr3=2e-2,
        constraint_weight=1.0,
    ):
        self.use_moire_component = False
        self._set_requires_grad(self.center_disk.template_raw, False)
        self._set_requires_grad(self.lat1.disk.template_raw, False)
        self._set_requires_grad(self.lat2.disk.template_raw, False)
        self._set_requires_grad(self.lat3.disk.template_raw, False)
        if self.moire_template_raw is not None:
            self._set_requires_grad(self.moire_template_raw, False)
        if self.moire_scale_raw is not None:
            self._set_requires_grad(self.moire_scale_raw, False)
        print("\n[Stage 1] geometry + intensity + background")
        self.fit(n_steps=stage1_steps, lr=lr1, constraint_weight=constraint_weight)

        self.use_moire_component = False
        self._set_requires_grad(self.lat1.disk.template_raw, True)
        self._set_requires_grad(self.lat2.disk.template_raw, True)
        self._set_requires_grad(self.lat3.disk.template_raw, True)
        print("\n[Stage 2] release parent spot templates")
        self.fit(n_steps=stage2_steps, lr=lr2, constraint_weight=constraint_weight)

        self.use_moire_component = True
        self._set_requires_grad(self.center_disk.template_raw, True)
        if self.moire_template_raw is not None:
            self._set_requires_grad(self.moire_template_raw, self.refine_moire_template)
        if self.moire_scale_raw is not None:
            self._set_requires_grad(self.moire_scale_raw, True)
        print("\n[Stage 3] final refinement + three-lattice moire term")
        self.fit(n_steps=stage3_steps, lr=lr3, constraint_weight=constraint_weight)

    def summary(self):
        out = self._current_parent_summary_numpy()
        out.update({
            "moire_scale": float(self.moire_scale_raw.detach().cpu()) if self.moire_scale_raw is not None else None,
            "final_loss": float(self.loss_history[-1]) if len(self.loss_history) else None,
        })
        moire_data = self.get_trained_moire_data()
        if moire_data is not None:
            out.update(moire_data)
            out["trained_moire_points_rc"] = moire_data.get("moire_active_points_rc", None)
        else:
            out["trained_moire_points_rc"] = None
        return out

    def plot(self, log_display=True, overlay=True, figsize=(15, 4)):
        pred = self.render_numpy()
        resid = self.image - pred

        def _disp(x):
            x = np.asarray(x, dtype=np.float32)
            if log_display:
                x = np.log1p(np.clip(x, 0, None))
            return x

        image_disp = _disp(self.image)
        pred_disp = _disp(pred)
        vmin_img = np.percentile(np.concatenate([image_disp.ravel(), pred_disp.ravel()]), 8)
        vmax_img = np.percentile(np.concatenate([image_disp.ravel(), pred_disp.ravel()]), 99.8)
        vmax_resid = max(np.percentile(np.abs(resid), 99), 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")

        axes[0].imshow(image_disp, cmap="Reds", vmin=vmin_img, vmax=vmax_img)
        axes[0].set_title("target")
        axes[0].axis("off")

        axes[1].imshow(pred_disp, cmap="Reds", vmin=vmin_img, vmax=vmax_img)
        axes[1].set_title("model")
        axes[1].axis("off")

        axes[2].imshow(resid, cmap="bwr", vmin=-vmax_resid, vmax=vmax_resid)
        axes[2].set_title("residual (target - model)")
        axes[2].axis("off")

        if overlay:
            s = self.summary()
            axes[1].scatter([s["origin_rc"][1]], [s["origin_rc"][0]], s=60, c="red", marker="+", linewidths=1.5)
            for key, color in [
                ("lattice1_points_rc", "cyan"),
                ("lattice2_points_rc", "yellow"),
                ("lattice3_points_rc", "lime"),
                ("moire12_points_rc", "orange"),
                ("moire13_points_rc", "magenta"),
                ("moire23_points_rc", "springgreen"),
                ("moire123_extra_points_rc", "black"),
            ]:
                pts = s.get(key, None)
                if pts is None or len(pts) == 0:
                    continue
                marker = "x" if "extra" in key else "o"
                if marker == "o":
                    axes[1].scatter(pts[:, 1], pts[:, 0], s=16, facecolors="none", edgecolors=color, linewidths=0.9)
                else:
                    axes[1].scatter(pts[:, 1], pts[:, 0], s=18, c=color, marker=marker, linewidths=0.9)

        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_history)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("fit loss")
        plt.tight_layout()
        plt.show()

# Three-lattice plots

def plot_refined_overlay_three(image, refined_data, figsize=(10, 10), title="refined overlay", log_display=True):
    def _disp(x):
        x = np.asarray(x, dtype=np.float32)
        if log_display:
            x = np.log1p(np.clip(x, 0, None))
        return x

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.imshow(_disp(image), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    origin_rc = refined_data["origin_rc"]
    u1_rc = refined_data["u1_rc"]
    v1_rc = refined_data["v1_rc"]
    u2_rc = refined_data["u2_rc"]
    v2_rc = refined_data["v2_rc"]
    u3_rc = refined_data["u3_rc"]
    v3_rc = refined_data["v3_rc"]

    c0 = (0.0, 1.0, 0.0)
    c1 = (1.0, 0.0, 0.0)
    c2 = (0.0, 0.8, 1.0)
    c3 = (0.0, 1.0, 0.5)
    c12 = (1.0, 0.5, 0.0)
    c13 = (0.6, 0.0, 1.0)
    c23 = (0.2, 0.9, 0.2)
    c123 = (0.0, 0.0, 0.0)

    ax.scatter([origin_rc[1]], [origin_rc[0]], s=90, marker="x", c=[c0], linewidths=1.5, zorder=10)

    def _draw_vec(vec, color):
        p = origin_rc + vec
        ax.plot([origin_rc[1], p[1]], [origin_rc[0], p[0]], color=color, linewidth=1.2, zorder=9)
        ax.scatter([p[1]], [p[0]], s=45, marker="o", facecolors="none", edgecolors=[color], linewidths=1.2, zorder=10)

    _draw_vec(u1_rc, c1)
    _draw_vec(v1_rc, c1)
    _draw_vec(u2_rc, c2)
    _draw_vec(v2_rc, c2)
    _draw_vec(u3_rc, c3)
    _draw_vec(v3_rc, c3)

    for key, color, label in [
        ("lattice1_points_rc", c1, "lattice1"),
        ("lattice2_points_rc", c2, "lattice2"),
        ("lattice3_points_rc", c3, "lattice3"),
        ("moire12_points_rc", c12, "moire12"),
        ("moire13_points_rc", c13, "moire13"),
        ("moire23_points_rc", c23, "moire23"),
        ("moire123_extra_points_rc", c123, "moire123_extra"),
    ]:
        pts = refined_data.get(key, None)
        if pts is None or len(pts) == 0:
            continue
        marker = "x" if "extra" in key else "o"
        if marker == "o":
            ax.scatter(pts[:, 1], pts[:, 0], s=16, marker=marker, facecolors="none", edgecolors=[color], linewidths=1.0, alpha=0.85, label=label)
        else:
            ax.scatter(pts[:, 1], pts[:, 0], s=24, marker=marker, c=[color], linewidths=1.0, alpha=0.9, label=label)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()
