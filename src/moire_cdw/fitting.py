"""Two-lattice fitting, initialization, geometry, and comparison plots."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from quantem.core.fitting.base import OriginND, RenderContext, AdditiveRenderModel
from quantem.core.fitting.diffraction import DiskTemplate, SyntheticDiskLattice
from quantem.core.fitting.background import DCBackground, GaussianBackground


# Shared initialization and templates

def robust_normalize(image, q_low=0.5, q_high=99.8):
    img = np.asarray(image, dtype=np.float32)
    lo = np.percentile(img, q_low)
    hi = np.percentile(img, q_high)
    img = np.clip(img - lo, 0, None)
    scale = max(hi - lo, 1e-8)
    img = img / scale
    return img.astype(np.float32)

def extract_patch(image, center_rc, radius_px):
    img = np.asarray(image, dtype=np.float32)
    r = int(np.round(center_rc[0]))
    c = int(np.round(center_rc[1]))
    rad = int(radius_px)

    r0 = max(0, r - rad)
    r1 = min(img.shape[0], r + rad + 1)
    c0 = max(0, c - rad)
    c1 = min(img.shape[1], c + rad + 1)

    patch = img[r0:r1, c0:c1].copy()

    out = np.zeros((2 * rad + 1, 2 * rad + 1), dtype=np.float32)
    rr0 = rad - (r - r0)
    cc0 = rad - (c - c0)
    out[rr0:rr0 + patch.shape[0], cc0:cc0 + patch.shape[1]] = patch
    return out

def make_template_from_peaks(image, peak_list_rc, radius_px, subtract_quantile=30.0):
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

def make_center_weight_mask(shape, center_rc, radius_px, center_weight=0.2):
    rr, cc = np.indices(shape, dtype=np.float32)
    r0, c0 = float(center_rc[0]), float(center_rc[1])
    d2 = (rr - r0) ** 2 + (cc - c0) ** 2
    mask = np.ones(shape, dtype=np.float32)
    mask[d2 <= float(radius_px) ** 2] = float(center_weight)
    return mask

def tile_points_bounded(origin_rc, u_rc, v_rc, shape, max_tile):
    origin_rc = np.asarray(origin_rc, dtype=float)
    u_rc = np.asarray(u_rc, dtype=float)
    v_rc = np.asarray(v_rc, dtype=float)

    a = np.arange(-max_tile, max_tile + 1, dtype=float)
    b = np.arange(-max_tile, max_tile + 1, dtype=float)
    aa, bb = np.meshgrid(a, b, indexing="ij")
    pts = origin_rc[None, None, :] + aa[..., None] * u_rc[None, None, :] + bb[..., None] * v_rc[None, None, :]
    pts = pts.reshape(-1, 2)

    keep = (
        (pts[:, 0] >= 0.0)
        & (pts[:, 0] <= (shape[0] - 1))
        & (pts[:, 1] >= 0.0)
        & (pts[:, 1] <= (shape[1] - 1))
    )
    return pts[keep]

def get_auto_init_from_moire(moire, vector_indices):
    peaks = np.asarray(moire.peaks, dtype=float)
    ind_origin, ind_u1, ind_v1, ind_u2, ind_v2 = [int(x) for x in vector_indices]

    origin_rc = peaks[ind_origin].copy()
    u1_rc = peaks[ind_u1].copy() - origin_rc
    v1_rc = peaks[ind_v1].copy() - origin_rc
    u2_rc = peaks[ind_u2].copy() - origin_rc
    v2_rc = peaks[ind_v2].copy() - origin_rc

    return {
        "origin_rc": origin_rc,
        "u1_rc": u1_rc,
        "v1_rc": v1_rc,
        "u2_rc": u2_rc,
        "v2_rc": v2_rc,
        "seed_peak_indices": [ind_origin, ind_u1, ind_v1, ind_u2, ind_v2],
    }

# Parent-lattice geometry

def build_moire_from_vectors(
    origin_rc,
    u1_rc,
    v1_rc,
    u2_rc,
    v2_rc,
    shape,
    max_tile=12,
    moire_exclude_radius_px=2.0,
    moire_parent_maxlen_px=None,
    moire_round_decimals=3,
):
    origin_rc = np.asarray(origin_rc, dtype=float)
    u1_rc = np.asarray(u1_rc, dtype=float)
    v1_rc = np.asarray(v1_rc, dtype=float)
    u2_rc = np.asarray(u2_rc, dtype=float)
    v2_rc = np.asarray(v2_rc, dtype=float)

    pts1 = tile_points_bounded(origin_rc, u1_rc, v1_rc, shape, max_tile=max_tile)
    pts2 = tile_points_bounded(origin_rc, u2_rc, v2_rc, shape, max_tile=max_tile)

    maxlen = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else float(0.5 * np.hypot(*shape))

    def _disp_set(u, v, maxlen):
        dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
        mt = int(np.ceil(maxlen / max(dmin, 1e-6))) + 1
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

    disp1, ab1 = _disp_set(u1_rc, v1_rc, maxlen)
    disp2, ab2 = _disp_set(u2_rc, v2_rc, maxlen)

    pts_moire = None
    if disp1.size and disp2.size:
        pts = origin_rc[None, None, :] + disp1[:, None, :] + disp2[None, :, :]
        pts = pts.reshape(-1, 2)

        keep_combo = ~(
            ((ab1[:, 0] == 0) & (ab1[:, 1] == 0))[:, None]
            | ((ab2[:, 0] == 0) & (ab2[:, 1] == 0))[None, :]
        ).ravel()
        pts = pts[keep_combo]

        keep_bounds = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] <= (shape[0] - 1))
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] <= (shape[1] - 1))
        )
        pts = pts[keep_bounds]

        if pts.size:
            pts = np.unique(np.round(pts, int(moire_round_decimals)), axis=0)

            if moire_exclude_radius_px > 0:
                parent_pts = np.vstack((origin_rc[None, :] + disp1, origin_rc[None, :] + disp2))
                parent_keep = (
                    (parent_pts[:, 0] >= 0.0)
                    & (parent_pts[:, 0] <= (shape[0] - 1))
                    & (parent_pts[:, 1] >= 0.0)
                    & (parent_pts[:, 1] <= (shape[1] - 1))
                )
                parent_pts = parent_pts[parent_keep]

                if parent_pts.size:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(parent_pts)
                    d, _ = tree.query(pts, k=1)
                    pts = pts[d >= float(moire_exclude_radius_px)]

            pts_moire = pts if pts.size else None

    return {
        "lattice1_points_rc": pts1,
        "lattice2_points_rc": pts2,
        "moire_points_rc": pts_moire,
        "moire_info": {
            "moire_parent_maxlen_px": maxlen,
            "moire_exclude_radius_px": float(moire_exclude_radius_px),
            "moire_round_decimals": int(moire_round_decimals),
            "moire_n_disp1": int(disp1.shape[0]),
            "moire_n_disp2": int(disp2.shape[0]),
        }
    }

# Explicit-moire geometry

def _build_explicit_moire_from_vectors(
    origin_rc,
    u1_rc,
    v1_rc,
    u2_rc,
    v2_rc,
    shape,
    max_tile=12,
    moire_exclude_radius_px=2.0,
    moire_parent_maxlen_px=None,
    moire_round_decimals=3,
    image_for_superlattice=None,  # kept only for API compatibility; not used
):
    """
    Generate parent-lattice points and moiré points using a self-contained
    geometric construction.

    Returns stable keys:
        - lattice1_points_rc
        - lattice2_points_rc
        - moire_points_rc
        - moire_info
    """
    origin_rc = np.asarray(origin_rc, dtype=float)
    u1_rc = np.asarray(u1_rc, dtype=float)
    v1_rc = np.asarray(v1_rc, dtype=float)
    u2_rc = np.asarray(u2_rc, dtype=float)
    v2_rc = np.asarray(v2_rc, dtype=float)

    # Parent lattices
    pts1 = tile_points_bounded(origin_rc, u1_rc, v1_rc, shape, max_tile=max_tile)
    pts2 = tile_points_bounded(origin_rc, u2_rc, v2_rc, shape, max_tile=max_tile)

    maxlen = float(moire_parent_maxlen_px) if moire_parent_maxlen_px is not None else float(0.5 * np.hypot(*shape))

    def _disp_set(u, v, maxlen):
        dmin = float(min(np.linalg.norm(u), np.linalg.norm(v)))
        mt = int(np.ceil(maxlen / max(dmin, 1e-6))) + 1
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

    disp1, ab1 = _disp_set(u1_rc, v1_rc, maxlen)
    disp2, ab2 = _disp_set(u2_rc, v2_rc, maxlen)

    pts_moire = None
    if disp1.size and disp2.size:
        pts = origin_rc[None, None, :] + disp1[:, None, :] + disp2[None, :, :]
        pts = pts.reshape(-1, 2)

        # remove trivial combinations involving zero displacement from either parent
        keep_combo = ~(
            ((ab1[:, 0] == 0) & (ab1[:, 1] == 0))[:, None]
            | ((ab2[:, 0] == 0) & (ab2[:, 1] == 0))[None, :]
        ).ravel()
        pts = pts[keep_combo]

        # keep points inside image bounds
        keep_bounds = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] <= (shape[0] - 1))
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] <= (shape[1] - 1))
        )
        pts = pts[keep_bounds]

        if pts.size:
            pts = np.unique(np.round(pts, int(moire_round_decimals)), axis=0)

            # remove points too close to parent-lattice points
            if moire_exclude_radius_px > 0:
                parent_pts = np.vstack((origin_rc[None, :] + disp1, origin_rc[None, :] + disp2))
                parent_keep = (
                    (parent_pts[:, 0] >= 0.0)
                    & (parent_pts[:, 0] <= (shape[0] - 1))
                    & (parent_pts[:, 1] >= 0.0)
                    & (parent_pts[:, 1] <= (shape[1] - 1))
                )
                parent_pts = parent_pts[parent_keep]

                if parent_pts.size:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(parent_pts)
                    d, _ = tree.query(pts, k=1)
                    pts = pts[d >= float(moire_exclude_radius_px)]

            pts_moire = pts if pts.size else None

    return {
        "lattice1_points_rc": pts1,
        "lattice2_points_rc": pts2,
        "moire_points_rc": pts_moire,
        "moire_info": {
            "moire_parent_maxlen_px": maxlen,
            "moire_exclude_radius_px": float(moire_exclude_radius_px),
            "moire_round_decimals": int(moire_round_decimals),
            "moire_n_disp1": int(disp1.shape[0]),
            "moire_n_disp2": int(disp2.shape[0]),
        },
    }

# Parent-lattice fitter

class TwoLatticeMoireFitter:
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
        self.dc_bg = None
        self.gaussian_bg = None

        self._origin_init = None
        self._origin_delta = None

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

    def build_from_auto_init(
        self,
        origin_rc,
        u1_rc,
        v1_rc,
        u2_rc,
        v2_rc,
        peak_seed_rc,
        center_radius_px=9,
        spot_radius_px=5,
        origin_delta_px=6.0,
        uv_delta_px=4.0,
        lattice_order=6,
        center_intensity=1.0,
        spot_intensity=0.15,
        gaussian_bg_sigma=70.0,
        gaussian_bg_intensity=0.03,
        dc_bg_intensity=0.005,
        refine_center_template=False,
        refine_spot_templates=True,
        tv_weight=1e-4,
        boundary_px=7.0,
    ):
        origin_rc = np.asarray(origin_rc, dtype=float)
        u1_rc = np.asarray(u1_rc, dtype=float)
        v1_rc = np.asarray(v1_rc, dtype=float)
        u2_rc = np.asarray(u2_rc, dtype=float)
        v2_rc = np.asarray(v2_rc, dtype=float)

        self._origin_init = origin_rc.copy()
        self._origin_delta = float(origin_delta_px)

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

        self.origin = OriginND(
            ndim=2,
            init=[float(origin_rc[0]), float(origin_rc[1])]
        ).to(self.device)

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

        disk1 = DiskTemplate.from_array(
            name="spot_disk1",
            array=spot_template,
            refine_all_pixels=bool(refine_spot_templates),
            normalize="none",
            intensity=0.0,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        disk1.set_origin(self.origin)

        disk2 = DiskTemplate.from_array(
            name="spot_disk2",
            array=spot_template,
            refine_all_pixels=bool(refine_spot_templates),
            normalize="none",
            intensity=0.0,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        disk2.set_origin(self.origin)

        self.lat1 = SyntheticDiskLattice(
            name="lat1",
            disk=disk1,
            u_row=(float(u1_rc[0]), float(uv_delta_px)),
            u_col=(float(u1_rc[1]), float(uv_delta_px)),
            v_row=(float(v1_rc[0]), float(uv_delta_px)),
            v_col=(float(v1_rc[1]), float(uv_delta_px)),
            u_max=int(lattice_order),
            v_max=int(lattice_order),
            per_disk_intensity=True,
            per_disk_slopes=False,
            intensity_0=spot_intensity,
            center_intensity_0=0.0,
            exclude_indices=None,
            boundary_px=float(boundary_px),
        )
        self.lat1.set_origin(self.origin)

        self.lat2 = SyntheticDiskLattice(
            name="lat2",
            disk=disk2,
            u_row=(float(u2_rc[0]), float(uv_delta_px)),
            u_col=(float(u2_rc[1]), float(uv_delta_px)),
            v_row=(float(v2_rc[0]), float(uv_delta_px)),
            v_col=(float(v2_rc[1]), float(uv_delta_px)),
            u_max=int(lattice_order),
            v_max=int(lattice_order),
            per_disk_intensity=True,
            per_disk_slopes=False,
            intensity_0=spot_intensity,
            center_intensity_0=0.0,
            exclude_indices=None,
            boundary_px=float(boundary_px),
        )
        self.lat2.set_origin(self.origin)

        self.dc_bg = DCBackground(
            name="dc_background",
            intensity=dc_bg_intensity,
        )

        self.gaussian_bg = GaussianBackground(
            name="gaussian_background",
            sigma=gaussian_bg_sigma,
            intensity=gaussian_bg_intensity,
        )
        self.gaussian_bg.set_origin(self.origin)

        components = [
            self.center_disk,
            self.dc_bg,
            self.gaussian_bg,
            self.lat1,
            self.lat2,
        ]
        self.model = AdditiveRenderModel(origin=self.origin, components=components).to(self.device)
        self.ctx = RenderContext(
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask_t,
        )

    def render_torch(self):
        if self.model is None:
            raise RuntimeError("Call build_from_auto_init() first.")
        return self.model(self.ctx)

    def render_numpy(self):
        with torch.no_grad():
            return self.render_torch().detach().cpu().numpy()

    def _enforce_origin_bounds(self):
        if self._origin_init is None or self._origin_delta is None:
            return
        with torch.no_grad():
            self.origin.coords[0].clamp_(
                float(self._origin_init[0] - self._origin_delta),
                float(self._origin_init[0] + self._origin_delta),
            )
            self.origin.coords[1].clamp_(
                float(self._origin_init[1] - self._origin_delta),
                float(self._origin_init[1] + self._origin_delta),
            )

    def fit(self, n_steps=150, lr=5e-2, constraint_weight=1.0, print_every=25):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        target_t = self._transform(self.target_t)
        hist = []

        for step in range(n_steps):
            optimizer.zero_grad()

            pred = self.render_torch()
            pred_t = self._transform(pred)

            data_loss = self._masked_mse(pred_t, target_t)
            reg_loss = self.model.total_constraint_loss(self.ctx)
            loss = data_loss + float(constraint_weight) * reg_loss

            loss.backward()
            optimizer.step()
            self.model.apply_hard_constraints(self.ctx)
            self._enforce_origin_bounds()

            loss_val = float(loss.detach().cpu())
            hist.append(loss_val)

            if print_every and ((step == 0) or ((step + 1) % print_every == 0) or (step == n_steps - 1)):
                print(
                    f"step {step+1:4d}/{n_steps} | "
                    f"loss={loss_val:.6e} | "
                    f"data={float(data_loss.detach().cpu()):.6e} | "
                    f"reg={float(reg_loss.detach().cpu()):.6e}"
                )

        self.loss_history.extend(hist)
        return hist

    def _set_requires_grad(self, tensor, enabled):
        tensor.requires_grad_(bool(enabled))

    def fit_staged(
        self,
        stage1_steps=40,
        stage2_steps=150,
        stage3_steps=100,
        lr1=2e-1,
        lr2=5e-2,
        lr3=2e-2,
        constraint_weight=1.0,
    ):
        # Stage 1: freeze templates, fit geometry/background/intensity first
        self._set_requires_grad(self.center_disk.template_raw, False)
        self._set_requires_grad(self.lat1.disk.template_raw, False)
        self._set_requires_grad(self.lat2.disk.template_raw, False)
        print("\n[Stage 1] geometry + intensity + background")
        self.fit(n_steps=stage1_steps, lr=lr1, constraint_weight=constraint_weight)

        # Stage 2: release spot templates
        self._set_requires_grad(self.lat1.disk.template_raw, True)
        self._set_requires_grad(self.lat2.disk.template_raw, True)
        print("\n[Stage 2] release spot templates")
        self.fit(n_steps=stage2_steps, lr=lr2, constraint_weight=constraint_weight)

        # Stage 3: optionally release center template too
        self._set_requires_grad(self.center_disk.template_raw, True)
        print("\n[Stage 3] final refinement")
        self.fit(n_steps=stage3_steps, lr=lr3, constraint_weight=constraint_weight)

    def summary(self):
        return {
            "origin_rc": self.origin.coords.detach().cpu().numpy().copy(),
            "u1_rc": np.array([
                float(self.lat1.u_row.detach().cpu()),
                float(self.lat1.u_col.detach().cpu()),
            ]),
            "v1_rc": np.array([
                float(self.lat1.v_row.detach().cpu()),
                float(self.lat1.v_col.detach().cpu()),
            ]),
            "u2_rc": np.array([
                float(self.lat2.u_row.detach().cpu()),
                float(self.lat2.u_col.detach().cpu()),
            ]),
            "v2_rc": np.array([
                float(self.lat2.v_row.detach().cpu()),
                float(self.lat2.v_col.detach().cpu()),
            ]),
            "final_loss": float(self.loss_history[-1]) if len(self.loss_history) else None,
        }

    def plot(self, log_display=True, overlay=True, figsize=(15, 4)):
        pred = self.render_numpy()
        resid = self.image - pred

        def _disp(x):
            if log_display:
                return np.log1p(np.clip(x, 0, None))
            return x

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(_disp(self.image), cmap="gray")
        axes[0].set_title("target")
        axes[0].axis("off")

        axes[1].imshow(_disp(pred), cmap="gray")
        axes[1].set_title("model")
        axes[1].axis("off")

        axes[2].imshow(resid, cmap="bwr")
        axes[2].set_title("residual (target - model)")
        axes[2].axis("off")

        if overlay:
            s = self.summary()
            refined_data = build_moire_from_vectors(
                origin_rc=s["origin_rc"],
                u1_rc=s["u1_rc"],
                v1_rc=s["v1_rc"],
                u2_rc=s["u2_rc"],
                v2_rc=s["v2_rc"],
                shape=self.shape,
                max_tile=12,
                moire_exclude_radius_px=2.0,
                moire_parent_maxlen_px=250,
            )
            pts1 = refined_data["lattice1_points_rc"]
            pts2 = refined_data["lattice2_points_rc"]
            axes[1].scatter([s["origin_rc"][1]], [s["origin_rc"][0]], s=60, c="red", marker="+", linewidths=1.5)

            if pts1 is not None and len(pts1):
                axes[1].scatter(pts1[:, 1], pts1[:, 0], s=18, facecolors="none", edgecolors="cyan", linewidths=1.0)
            if pts2 is not None and len(pts2):
                axes[1].scatter(pts2[:, 1], pts2[:, 0], s=18, facecolors="none", edgecolors="yellow", linewidths=1.0)

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

# Explicit-moire fitter

class ExplicitMoireFitter:
    """
    Two parent lattices are trained directly. Moiré points are generated from the
    current trained parent lattices using the notebook's own geometric build_moire_from_vectors(...), and a weak explicit moiré intensity term is added in
    Stage 3 so the visible moiré spots also contribute to the loss.

    Important:
    - Parent lattices (lat1, lat2) remain the main trainable geometric model.
    - Moiré points are NOT generated from a fake third SyntheticDiskLattice.
      They are generated by the notebook's own geometric moiré-construction function.
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
        self.dc_bg = None
        self.gaussian_bg = None

        # Explicit moiré-intensity term (generated from build_moire_from_vectors)
        self.moire_template_raw = None
        self.moire_scale_raw = None
        self.use_moire_component = False

        self._origin_init = None
        self._origin_delta = None

        # moiré-generation settings
        self.moire_max_tile = 12
        self.moire_exclude_radius_px = 2.0
        self.moire_parent_maxlen_px = 180.0
        self.moire_round_decimals = 3

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
            "u1_rc": np.array([
                float(self.lat1.u_row.detach().cpu()),
                float(self.lat1.u_col.detach().cpu()),
            ]),
            "v1_rc": np.array([
                float(self.lat1.v_row.detach().cpu()),
                float(self.lat1.v_col.detach().cpu()),
            ]),
            "u2_rc": np.array([
                float(self.lat2.u_row.detach().cpu()),
                float(self.lat2.u_col.detach().cpu()),
            ]),
            "v2_rc": np.array([
                float(self.lat2.v_row.detach().cpu()),
                float(self.lat2.v_col.detach().cpu()),
            ]),
        }

    def get_trained_moire_points(self):
        if self.origin is None or self.lat1 is None or self.lat2 is None:
            return None
        s = self._current_parent_summary_numpy()
        data = _build_explicit_moire_from_vectors(
            origin_rc=s["origin_rc"],
            u1_rc=s["u1_rc"],
            v1_rc=s["v1_rc"],
            u2_rc=s["u2_rc"],
            v2_rc=s["v2_rc"],
            shape=self.shape,
            max_tile=self.moire_max_tile,
            moire_exclude_radius_px=self.moire_exclude_radius_px,
            moire_parent_maxlen_px=self.moire_parent_maxlen_px,
            moire_round_decimals=self.moire_round_decimals,

        )
        return data.get("moire_points_rc", None)

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

    def build_from_auto_init(
        self,
        origin_rc,
        u1_rc,
        v1_rc,
        u2_rc,
        v2_rc,
        peak_seed_rc,
        center_radius_px=9,
        spot_radius_px=5,
        origin_delta_px=6.0,
        uv_delta_px=4.0,
        lattice_order=6,
        center_intensity=1.0,
        spot_intensity=0.15,
        moire_intensity=0.02,
        moire_max_tile=12,
        moire_exclude_radius_px=2.0,
        moire_parent_maxlen_px=180.0,
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

        self._origin_init = origin_rc.copy()
        self._origin_delta = float(origin_delta_px)

        self.moire_max_tile = int(moire_max_tile)
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

        # Keep moiré template conservative: start from the same local spot template,
        # but with weak intensity. The actual moiré POINTS are generated from the
        # own moiré-construction function, not from a fake third lattice basis.
        moire_template = spot_template.copy()

        self.origin = OriginND(
            ndim=2,
            init=[float(origin_rc[0]), float(origin_rc[1])]
        ).to(self.device)

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

        disk1 = DiskTemplate.from_array(
            name="spot_disk1",
            array=spot_template,
            refine_all_pixels=bool(refine_spot_templates),
            normalize="none",
            intensity=0.0,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        disk1.set_origin(self.origin)

        disk2 = DiskTemplate.from_array(
            name="spot_disk2",
            array=spot_template,
            refine_all_pixels=bool(refine_spot_templates),
            normalize="none",
            intensity=0.0,
            constraint_params={
                "force_center": True,
                "force_positive": True,
                "tv_weight": float(tv_weight),
            },
        )
        disk2.set_origin(self.origin)

        self.lat1 = SyntheticDiskLattice(
            name="lat1",
            disk=disk1,
            u_row=(float(u1_rc[0]), float(uv_delta_px)),
            u_col=(float(u1_rc[1]), float(uv_delta_px)),
            v_row=(float(v1_rc[0]), float(uv_delta_px)),
            v_col=(float(v1_rc[1]), float(uv_delta_px)),
            u_max=int(lattice_order),
            v_max=int(lattice_order),
            per_disk_intensity=True,
            per_disk_slopes=False,
            intensity_0=spot_intensity,
            center_intensity_0=0.0,
            exclude_indices=None,
            boundary_px=float(boundary_px),
        )
        self.lat1.set_origin(self.origin)

        self.lat2 = SyntheticDiskLattice(
            name="lat2",
            disk=disk2,
            u_row=(float(u2_rc[0]), float(uv_delta_px)),
            u_col=(float(u2_rc[1]), float(uv_delta_px)),
            v_row=(float(v2_rc[0]), float(uv_delta_px)),
            v_col=(float(v2_rc[1]), float(uv_delta_px)),
            u_max=int(lattice_order),
            v_max=int(lattice_order),
            per_disk_intensity=True,
            per_disk_slopes=False,
            intensity_0=spot_intensity,
            center_intensity_0=0.0,
            exclude_indices=None,
            boundary_px=float(boundary_px),
        )
        self.lat2.set_origin(self.origin)

        self.dc_bg = DCBackground(
            name="dc_background",
            intensity=dc_bg_intensity,
        )

        self.gaussian_bg = GaussianBackground(
            name="gaussian_background",
            sigma=gaussian_bg_sigma,
            intensity=gaussian_bg_intensity,
        )
        self.gaussian_bg.set_origin(self.origin)

        components = [
            self.center_disk,
            self.dc_bg,
            self.gaussian_bg,
            self.lat1,
            self.lat2,
        ]
        self.model = AdditiveRenderModel(origin=self.origin, components=components).to(self.device)
        self.ctx = RenderContext(
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask_t,
        )

        self.moire_template_raw = torch.nn.Parameter(
            torch.as_tensor(moire_template, device=self.device, dtype=self.dtype)
        )
        self.moire_scale_raw = torch.nn.Parameter(
            torch.tensor(float(moire_intensity), device=self.device, dtype=self.dtype)
        )
        self.moire_template_raw.requires_grad_(bool(refine_moire_template))
        self.moire_scale_raw.requires_grad_(True)

        # Start conservatively: enable explicit moiré only in the final stage.
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
            self.origin.coords[0].clamp_(
                float(self._origin_init[0] - self._origin_delta),
                float(self._origin_init[0] + self._origin_delta),
            )
            self.origin.coords[1].clamp_(
                float(self._origin_init[1] - self._origin_delta),
                float(self._origin_init[1] + self._origin_delta),
            )

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
        stage2_steps=150,
        stage3_steps=100,
        lr1=2e-1,
        lr2=5e-2,
        lr3=2e-2,
        constraint_weight=1.0,
    ):
        # Stage 1: freeze templates, fit geometry/background/intensity first
        self.use_moire_component = False
        self._set_requires_grad(self.center_disk.template_raw, False)
        self._set_requires_grad(self.lat1.disk.template_raw, False)
        self._set_requires_grad(self.lat2.disk.template_raw, False)
        if self.moire_template_raw is not None:
            self._set_requires_grad(self.moire_template_raw, False)
        if self.moire_scale_raw is not None:
            self._set_requires_grad(self.moire_scale_raw, False)
        print("\n[Stage 1] geometry + intensity + background")
        self.fit(n_steps=stage1_steps, lr=lr1, constraint_weight=constraint_weight)

        # Stage 2: release parent-lattice spot templates
        self.use_moire_component = False
        self._set_requires_grad(self.lat1.disk.template_raw, True)
        self._set_requires_grad(self.lat2.disk.template_raw, True)
        print("\n[Stage 2] release parent spot templates")
        self.fit(n_steps=stage2_steps, lr=lr2, constraint_weight=constraint_weight)

        # Stage 3: enable explicit moiré term generated from own moiré-construction function
        self.use_moire_component = True
        self._set_requires_grad(self.center_disk.template_raw, True)
        if self.moire_template_raw is not None:
            self._set_requires_grad(self.moire_template_raw, self.refine_moire_template)
        if self.moire_scale_raw is not None:
            self._set_requires_grad(self.moire_scale_raw, True)
        print("\n[Stage 3] final refinement + own function-generated moiré term")
        self.fit(n_steps=stage3_steps, lr=lr3, constraint_weight=constraint_weight)

    def summary(self):
        out = {
            "origin_rc": self.origin.coords.detach().cpu().numpy().copy(),
            "u1_rc": np.array([
                float(self.lat1.u_row.detach().cpu()),
                float(self.lat1.u_col.detach().cpu()),
            ]),
            "v1_rc": np.array([
                float(self.lat1.v_row.detach().cpu()),
                float(self.lat1.v_col.detach().cpu()),
            ]),
            "u2_rc": np.array([
                float(self.lat2.u_row.detach().cpu()),
                float(self.lat2.u_col.detach().cpu()),
            ]),
            "v2_rc": np.array([
                float(self.lat2.v_row.detach().cpu()),
                float(self.lat2.v_col.detach().cpu()),
            ]),
            "moire_scale": float(self.moire_scale_raw.detach().cpu()) if self.moire_scale_raw is not None else None,
            "final_loss": float(self.loss_history[-1]) if len(self.loss_history) else None,
        }

        ptsm = self.get_trained_moire_points()
        out["trained_moire_points_rc"] = ptsm
        return out

    def plot(self, log_display=True, overlay=True, figsize=(15, 4)):
        pred = self.render_numpy()
        resid = self.image - pred

        def _disp(x):
            if log_display:
                return np.log1p(np.clip(x, 0, None))
            return x

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(_disp(self.image), cmap="gray")
        axes[0].set_title("target")
        axes[0].axis("off")

        axes[1].imshow(_disp(pred), cmap="gray")
        axes[1].set_title("model")
        axes[1].axis("off")

        axes[2].imshow(resid, cmap="bwr")
        axes[2].set_title("residual (target - model)")
        axes[2].axis("off")

        if overlay:
            s = self.summary()
            refined_data = _build_explicit_moire_from_vectors(
                origin_rc=s["origin_rc"],
                u1_rc=s["u1_rc"],
                v1_rc=s["v1_rc"],
                u2_rc=s["u2_rc"],
                v2_rc=s["v2_rc"],
                shape=self.shape,
                max_tile=self.moire_max_tile,
                moire_exclude_radius_px=self.moire_exclude_radius_px,
                moire_parent_maxlen_px=self.moire_parent_maxlen_px,
                moire_round_decimals=self.moire_round_decimals,

            )
            pts1 = refined_data["lattice1_points_rc"]
            pts2 = refined_data["lattice2_points_rc"]
            ptsm = refined_data["moire_points_rc"]

            axes[1].scatter([s["origin_rc"][1]], [s["origin_rc"][0]], s=60, c="red", marker="+", linewidths=1.5)

            if pts1 is not None and len(pts1):
                axes[1].scatter(pts1[:, 1], pts1[:, 0], s=18, facecolors="none", edgecolors="cyan", linewidths=1.0)
            if pts2 is not None and len(pts2):
                axes[1].scatter(pts2[:, 1], pts2[:, 0], s=18, facecolors="none", edgecolors="yellow", linewidths=1.0)
            if ptsm is not None and len(ptsm):
                axes[1].scatter(ptsm[:, 1], ptsm[:, 0], s=16, facecolors="none", edgecolors="magenta", linewidths=1.0)

        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(3, len(self.loss_history) + 1), self.loss_history[2:])
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("fit loss")
        plt.tight_layout()
        plt.show()

# Comparison metrics and plots

def compute_fit_metrics(target, pred, mask=None):
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    if mask is None:
        w = np.ones_like(target, dtype=np.float32)
    else:
        w = np.asarray(mask, dtype=np.float32)

    wsum = np.clip(w.sum(), 1e-8, None)
    diff = target - pred

    rmse = np.sqrt((w * diff**2).sum() / wsum)

    t_log = np.log1p(np.clip(target, 0, None))
    p_log = np.log1p(np.clip(pred, 0, None))
    diff_log = t_log - p_log
    rmse_log = np.sqrt((w * diff_log**2).sum() / wsum)

    mae = (w * np.abs(diff)).sum() / wsum

    t_mean = (w * target).sum() / wsum
    p_mean = (w * pred).sum() / wsum
    num = (w * (target - t_mean) * (pred - p_mean)).sum()
    den = np.sqrt((w * (target - t_mean)**2).sum() * (w * (pred - p_mean)**2).sum()) + 1e-8
    corr = num / den

    return {
        "rmse_linear": float(rmse),
        "rmse_log1p": float(rmse_log),
        "mae_linear": float(mae),
        "weighted_corr": float(corr),
    }

def compare_vector_sets(initial_summary, refined_summary):
    rows = []
    keys = ["origin_rc", "u1_rc", "v1_rc", "u2_rc", "v2_rc"]
    for key in keys:
        before = np.asarray(initial_summary[key], dtype=float)
        after = np.asarray(refined_summary[key], dtype=float)
        delta = after - before
        rows.append((key, before, after, delta, float(np.linalg.norm(delta))))
    return rows

def plot_before_after_overlays(image, initial_data, refined_data, figsize=(15, 7), log_display=True):
    def _disp(x):
        if log_display:
            return np.log1p(np.clip(x, 0, None))
        return x

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, title in [
        (axes[0], initial_data, "Before fitting (auto-found)"),
        (axes[1], refined_data, "After fitting (refined)"),
    ]:
        ax.imshow(_disp(image), cmap="gray")
        ax.axis("off")
        ax.set_title(title)

        o = np.asarray(data["origin_rc"])
        ax.scatter([o[1]], [o[0]], s=70, c="red", marker="+", linewidths=1.8)

        pts1 = data["lattice1_points_rc"]
        pts2 = data["lattice2_points_rc"]
        mpts = data.get("trained_moire_points_rc", None)
        if mpts is None:
            mpts = data["moire_points_rc"]

        if pts1 is not None and len(pts1):
            ax.scatter(pts1[:, 1], pts1[:, 0], s=16, facecolors="none", edgecolors="cyan", linewidths=0.9, label="lattice 1")
        if pts2 is not None and len(pts2):
            ax.scatter(pts2[:, 1], pts2[:, 0], s=16, facecolors="none", edgecolors="yellow", linewidths=0.9, label="lattice 2")
        if mpts is not None and len(mpts):
            ax.scatter(mpts[:, 1], mpts[:, 0], s=20, c="lime", marker="x", linewidths=0.8, label="moire points (superlattice-generated)")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_before_after_models(image, pred_before, pred_after, figsize=(16, 10), log_display=True):
    def _disp(x):
        x = np.asarray(x, dtype=np.float32)
        if log_display:
            x = np.log1p(np.clip(x, 0, None))
        return x

    image = np.asarray(image, dtype=np.float32)
    pred_before = np.asarray(pred_before, dtype=np.float32)
    pred_after = np.asarray(pred_after, dtype=np.float32)

    image_disp = _disp(image)
    pred_before_disp = _disp(pred_before)
    pred_after_disp = _disp(pred_after)

    resid_before = image - pred_before
    resid_after = image - pred_after

    # Use a common target/model range and clip the darkest background for visibility.
    all_img_vals = np.concatenate([
        image_disp.ravel(),
        pred_before_disp.ravel(),
        pred_after_disp.ravel()
    ])
    vmin_img = np.percentile(all_img_vals, 8)
    vmax_img = np.percentile(all_img_vals, 99.8)

    # Use a shared symmetric residual color scale.
    v_before = np.percentile(np.abs(resid_before), 99)
    v_after = np.percentile(np.abs(resid_after), 99)
    v_resid = max(v_before, v_after, 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=figsize, facecolor="white")

    cmap_img = "Reds"

    # -------- Row 1: before fitting --------
    im00 = axes[0, 0].imshow(image_disp, cmap=cmap_img, vmin=vmin_img, vmax=vmax_img)
    axes[0, 0].set_title("Target image")
    axes[0, 0].axis("off")
    cbar00 = plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar00.set_label("Display intensity")

    im01 = axes[0, 1].imshow(pred_before_disp, cmap=cmap_img, vmin=vmin_img, vmax=vmax_img)
    axes[0, 1].set_title("Model before fitting")
    axes[0, 1].axis("off")
    cbar01 = plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar01.set_label("Display intensity")

    im02 = axes[0, 2].imshow(resid_before, cmap="bwr", vmin=-v_resid, vmax=v_resid)
    axes[0, 2].set_title("Residual before fitting\n(target - model)")
    axes[0, 2].axis("off")
    cbar02 = plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar02.set_label("Residual intensity\nred: target > model\nblue: target < model")

    # -------- Row 2: after fitting --------
    im10 = axes[1, 0].imshow(image_disp, cmap=cmap_img, vmin=vmin_img, vmax=vmax_img)
    axes[1, 0].set_title("Target image")
    axes[1, 0].axis("off")
    cbar10 = plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar10.set_label("Display intensity")

    im11 = axes[1, 1].imshow(pred_after_disp, cmap=cmap_img, vmin=vmin_img, vmax=vmax_img)
    axes[1, 1].set_title("Model after fitting")
    axes[1, 1].axis("off")
    cbar11 = plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar11.set_label("Display intensity")

    im12 = axes[1, 2].imshow(resid_after, cmap="bwr", vmin=-v_resid, vmax=v_resid)
    axes[1, 2].set_title("Residual after fitting\n(target - model)")
    axes[1, 2].axis("off")
    cbar12 = plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar12.set_label("Residual intensity\nred: target > model\nblue: target < model")

    plt.tight_layout()
    plt.show()

def plot_refined_overlay(image, refined_data, figsize=(10, 10), title="refined overlay"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap="gray")
    ax.set_title(title)

    origin_rc = refined_data["origin_rc"]
    u1_rc = refined_data["u1_rc"]
    v1_rc = refined_data["v1_rc"]
    u2_rc = refined_data["u2_rc"]
    v2_rc = refined_data["v2_rc"]

    c0 = (0.0, 1.0, 0.0)
    c1 = (1.0, 0.0, 0.0)
    c2 = (0.0, 0.8, 1.0)
    cm = (0.8, 0.0, 1.0)

    ax.scatter([origin_rc[1]], [origin_rc[0]], s=90, marker="x", c=[c0], linewidths=1.5, zorder=8)

    def _draw_vec(vec, color):
        p = origin_rc + vec
        ax.plot([origin_rc[1], p[1]], [origin_rc[0], p[0]], color=color, linewidth=1.2, zorder=7)
        ax.scatter([p[1]], [p[0]], s=45, marker="o", facecolors="none", edgecolors=[color], linewidths=1.2, zorder=8)

    _draw_vec(u1_rc, c1)
    _draw_vec(v1_rc, c1)
    _draw_vec(u2_rc, c2)
    _draw_vec(v2_rc, c2)

    pts1 = refined_data["lattice1_points_rc"]
    pts2 = refined_data["lattice2_points_rc"]
    ptsm = refined_data["moire_points_rc"]

    if pts1 is not None and len(pts1):
        ax.scatter(pts1[:, 1], pts1[:, 0], s=20, marker="o", facecolors="none", edgecolors=[c1], linewidths=1.0, alpha=0.8)
    if pts2 is not None and len(pts2):
        ax.scatter(pts2[:, 1], pts2[:, 0], s=20, marker="o", facecolors="none", edgecolors=[c2], linewidths=1.0, alpha=0.8)
    if ptsm is not None and len(ptsm):
        ax.scatter(ptsm[:, 1], ptsm[:, 0], s=14, marker="o", facecolors="none", edgecolors=[cm], linewidths=1.0, alpha=0.9)

    plt.tight_layout()
    plt.show()

# Optional parent-only stage plots

def plot_stage_result(image, pred, stage_name="Stage", figsize=(15, 4), log_display=True):
    def _disp(x):
        if log_display:
            return np.log1p(np.clip(x, 0, None))
        return x

    image = np.asarray(image, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    image_disp = _disp(image)
    pred_disp = _disp(pred)
    resid = image - pred

    # Use a shared intensity range.
    vmin_int = min(image_disp.min(), pred_disp.min())
    vmax_int = max(image_disp.max(), pred_disp.max())

    # Use a symmetric residual color scale.
    v_resid = max(np.percentile(np.abs(resid), 99), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    intensity_cmap = "inferno"   # Alternatives: "magma", "viridis", or "plasma".

    im0 = axes[0].imshow(image_disp, cmap=intensity_cmap, vmin=vmin_int, vmax=vmax_int)
    axes[0].set_title(f"{stage_name}: Target image")
    axes[0].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Display intensity")

    im1 = axes[1].imshow(pred_disp, cmap=intensity_cmap, vmin=vmin_int, vmax=vmax_int)
    axes[1].set_title(f"{stage_name}: Model")
    axes[1].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Display intensity")

    im2 = axes[2].imshow(resid, cmap="bwr", vmin=-v_resid, vmax=v_resid)
    axes[2].set_title(f"{stage_name}: Residual\n(target - model)")
    axes[2].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Residual intensity\nred: target > model\nblue: target < model")

    plt.tight_layout()
    plt.show()

def fit_staged_with_stage_plots(
    self,
    stage1_steps=40,
    stage2_steps=150,
    stage3_steps=100,
    lr1=2e-1,
    lr2=5e-2,
    lr3=2e-2,
    constraint_weight=1.0,
    show_before=True,
    show_stage1=True,
    show_stage2=True,
    show_stage3=True,
):
    # -------- Before fitting --------
    self.before_render = self.render_numpy()
    self.before_summary = self.summary()

    if show_before:
        print("\n--- Plot before fitting ---")
        plot_stage_result(
            self.image,
            self.before_render,
            stage_name="Before fitting",
            figsize=(15, 4),
            log_display=True,
        )

    # -------- Stage 1: freeze templates, fit geometry/background/intensity --------
    self._set_requires_grad(self.center_disk.template_raw, False)
    self._set_requires_grad(self.lat1.disk.template_raw, False)
    self._set_requires_grad(self.lat2.disk.template_raw, False)

    print("\n[Stage 1] geometry + intensity + background")
    self.fit(n_steps=stage1_steps, lr=lr1, constraint_weight=constraint_weight)

    self.stage1_render = self.render_numpy()
    self.stage1_summary = self.summary()

    if show_stage1:
        print("\n--- Plot after Stage 1 ---")
        plot_stage_result(
            self.image,
            self.stage1_render,
            stage_name="Stage 1",
            figsize=(15, 4),
            log_display=True,
        )

    # -------- Stage 2: release spot templates --------
    self._set_requires_grad(self.lat1.disk.template_raw, True)
    self._set_requires_grad(self.lat2.disk.template_raw, True)

    print("\n[Stage 2] release spot templates")
    self.fit(n_steps=stage2_steps, lr=lr2, constraint_weight=constraint_weight)

    self.stage2_render = self.render_numpy()
    self.stage2_summary = self.summary()

    if show_stage2:
        print("\n--- Plot after Stage 2 ---")
        plot_stage_result(
            self.image,
            self.stage2_render,
            stage_name="Stage 2",
            figsize=(15, 4),
            log_display=True,
        )

    # -------- Stage 3: final refinement --------
    self._set_requires_grad(self.center_disk.template_raw, True)

    print("\n[Stage 3] final refinement")
    self.fit(n_steps=stage3_steps, lr=lr3, constraint_weight=constraint_weight)

    self.stage3_render = self.render_numpy()
    self.stage3_summary = self.summary()

    if show_stage3:
        print("\n--- Plot after Stage 3 ---")
        plot_stage_result(
            self.image,
            self.stage3_render,
            stage_name="Stage 3",
            figsize=(15, 4),
            log_display=True,
        )
