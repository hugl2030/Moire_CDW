"""Independent post-fit calibration, preserving the original affine transform.

The reference lengths are real-space d spacings; ``im.sampling[0]`` is the
reciprocal-space step per binned pixel, in the inverse of the same length unit.
The original function prints diagnostics and does not modify the image/peaks.
"""

import numpy as np
import matplotlib.pyplot as plt


def calibration_from_vectors(
    origin_rc, u1_rc, v1_rc, u2_rc, v2_rc, *, sampling,
    u1_calibrate=0.219, v1_calibrate=0.219, angle1_calibrate=60,
    return_info=True, plot_vectors=True,
):
    """Pass fitted vectors to the unchanged :func:`calibration` function.

    ``sampling`` must be the *binned* diffraction sampling (scalar or [row, col]).
    As in the original algorithm, only its first value is used. This wrapper
    does not rescale an image or update a fitter. With a missing reference
    (None), the original function returns only the uncalibrated twist angle.
    """
    from types import SimpleNamespace

    origin = np.asarray(origin_rc, dtype=float).reshape(2)
    vectors = np.asarray([u1_rc, v1_rc, u2_rc, v2_rc], dtype=float)
    if vectors.shape != (4, 2) or not np.isfinite(vectors).all() or not np.isfinite(origin).all():
        raise ValueError("origin and vectors must contain finite (row, col) coordinates")
    for pair in (vectors[:2], vectors[2:]):
        if abs(np.linalg.det(pair)) < 1e-12:
            raise ValueError("Lattice basis is singular (zero-length or collinear vectors)")
    step = np.asarray(sampling, dtype=float).reshape(-1)
    if step.size not in (1, 2) or not np.isfinite(step).all() or np.any(step <= 0):
        raise ValueError("sampling must be a positive scalar or [row, col] pair")
    peaks = np.vstack([origin, origin + vectors])
    return calibration(
        peaks, [0, 1, 2, 3, 4], SimpleNamespace(sampling=step),
        u1_calibrate=u1_calibrate, v1_calibrate=v1_calibrate,
        angle1_calibrate=angle1_calibrate, return_info=return_info,
        plot_vectors=plot_vectors,
    )

def calibration(
    peaks,
    vector_indices,
    im,
    u1_calibrate = 0.219,
    v1_calibrate = 0.219,
    angle1_calibrate = 60,
    return_info = False,
    plot_vectors=True,
):
    origin = peaks[vector_indices[0]]
    u1 = peaks[vector_indices[1]] - origin
    v1 = peaks[vector_indices[2]] - origin
    u2 = peaks[vector_indices[3]] - origin
    v2 = peaks[vector_indices[4]] - origin

    def calculate_vector_angles(u1, v1):
        u1 = np.asarray(u1, dtype=float)
        v1 = np.asarray(v1, dtype=float)

        cos_theta = np.dot(u1, v1) / (np.linalg.norm(u1) * np.linalg.norm(v1))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # [0, pi]

        return float(np.degrees(theta))

    def calculate_length_ratio(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nv == 0:
            return float("nan")
        return nu / nv

    def find_twist_angle(u1, v1, u2, v2):
        angle_vector1 = calculate_vector_angles(u1, v1)
        angle_vector2 = calculate_vector_angles(u2, v2)
        angle_u1_v2 = calculate_vector_angles(u1, v2)
        angle_u2_v1 = calculate_vector_angles(u2, v1)
        angle_u1_u2 = calculate_vector_angles(u1, u2)
        angle_v1_v2 = calculate_vector_angles(v1, v2)
        angles_4 = [angle_u1_v2, angle_u2_v1, angle_u1_u2, angle_v1_v2]
        print(angle_vector1, angle_vector2, angles_4)

        period = min(angle_vector1, angle_vector2)          # min(angle_vector1, angle_vector2)
        half = 0.5 * period                                 # min(angle_vector1/2, angle_vector2/2)

        candidates = []
        for a in angles_4:
            for k in (0, 1, 2):
                x = a - k * period
                if 0.0 <= x <= period:
                    twist = x if x <= half else (period - x)
                    candidates.append(twist)

        if not candidates:
            x = angles_4[0] % period
            twist_angle = x if x <= half else (period - x)
        else:
            twist_angle = min(candidates)
        return float(twist_angle)

    twist_angle_before_calibrate = find_twist_angle(u1, v1, u2, v2)
    ratio_before_1 = calculate_length_ratio(u1, u2)
    ratio_before_2 = calculate_length_ratio(v1, v2)

    if u1_calibrate == None or v1_calibrate == None or angle1_calibrate == None:
        print(f"[Calibration missing] twist angle = {twist_angle_before_calibrate:.3f}°")
        print(f"[Calibration missing] |u1|/|u2| = {ratio_before_1:.3f},  |v1|/|v2| = {ratio_before_2:.3f}")
        return twist_angle_before_calibrate

    else:
        if hasattr(im, "sampling"):
            px = float(im.sampling[0])
        else:
            px = 0.01

        u1_length_calibrate = float((1.0 / u1_calibrate) / px)
        v1_length_calibrate = float((1.0 / v1_calibrate) / px)

        angle_vector1 = calculate_vector_angles(u1, v1)  # degrees
        ang_a = float(angle1_calibrate)
        ang_b = 180.0 - float(angle1_calibrate)
        angle1_final = ang_a if abs(angle_vector1 - ang_a) <= abs(angle_vector1 - ang_b) else ang_b


        u1_ref = np.array([0.0, u1_length_calibrate], dtype=float)

        theta = np.deg2rad(angle1_final)
        v1_ref = np.array([v1_length_calibrate * np.sin(theta),
                           v1_length_calibrate * np.cos(theta)], dtype=float)


        M_meas = np.column_stack((u1.astype(float), v1.astype(float)))      # 2x2
        M_ref  = np.column_stack((u1_ref, v1_ref))                          # 2x2

        det = float(np.linalg.det(M_meas))
        if abs(det) < 1e-12:
            raise ValueError("Measured basis [u1, v1] is nearly singular (collinear). Cannot calibrate.")

        A = M_ref @ np.linalg.inv(M_meas)

        def _apply(A, vec):
            vec = np.asarray(vec, dtype=float)
            return (A @ vec.reshape(2, 1)).ravel()

        u1_cal = _apply(A, u1)
        v1_cal = _apply(A, v1)
        u2_cal = _apply(A, u2)
        v2_cal = _apply(A, v2)
        ratio_after_1 = calculate_length_ratio(u1_cal, u2_cal)
        ratio_after_2 = calculate_length_ratio(v1_cal, v2_cal)

        if plot_vectors:
            all_vec = np.vstack([u1, v1, u2, v2, u1_cal, v1_cal, u2_cal, v2_cal]).astype(float)
            max_abs = float(np.max(np.abs(all_vec))) if all_vec.size else 1.0
            lim = 1.1 * max_abs + 1e-6

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axhline(0, linewidth=1)
            ax.axvline(0, linewidth=1)

            c_lat1 = "C0"  # u1,v1
            c_lat2 = "C1"  # u2,v2

            def draw_vec(vec, color, label, ls="-", alpha=1.0):
                vec = np.asarray(vec, float)


                x = vec[1]
                y = -vec[0]

                ax.arrow(
                    0, 0, x, y,
                    head_width=0.03 * lim, head_length=0.05 * lim,
                    length_includes_head=True,
                    linewidth=2,
                    linestyle=ls,
                    color=color,
                    alpha=alpha,
                )
                ax.text(x * 1.03, y * 1.03, label, color=color, fontsize=10)


            draw_vec(u1, c_lat1, "u1 (raw)", ls="--", alpha=0.75)
            draw_vec(v1, c_lat1, "v1 (raw)", ls="--", alpha=0.75)
            draw_vec(u2, c_lat2, "u2 (raw)", ls="--", alpha=0.75)
            draw_vec(v2, c_lat2, "v2 (raw)", ls="--", alpha=0.75)


            draw_vec(u1_cal, c_lat1, "u1 (cal)", ls="-", alpha=1.0)
            draw_vec(v1_cal, c_lat1, "v1 (cal)", ls="-", alpha=1.0)
            draw_vec(u2_cal, c_lat2, "u2 (cal)", ls="-", alpha=1.0)
            draw_vec(v2_cal, c_lat2, "v2 (cal)", ls="-", alpha=1.0)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            ax.set_xlabel("col (x)")
            ax.set_ylabel("-row (y)")
            ax.set_title("Vectors before/after calibration (row sign flipped)")
            ax.grid(True, linestyle=":", alpha=0.5)

            plt.show()

        angle_after_1 = calculate_vector_angles(u1_cal, v1_cal)
        angle_after_2 = calculate_vector_angles(u2_cal, v2_cal)

        twist_angle_after_calibrate = find_twist_angle(u1_cal, v1_cal, u2_cal, v2_cal)
        print(f"[Calibration] angle_vector1(before) = {angle_vector1:.3f}°")
        print(f"[Calibration] angle1_calibrate candidates = {ang_a:.3f}° or {ang_b:.3f}° -> chosen {angle1_final:.3f}°")
        print(f"[Calibration] |u1_ref|={np.linalg.norm(u1_ref):.3f}px, |v1_ref|={np.linalg.norm(v1_ref):.3f}px")
        print(f"[Calibration] A=\n{A}")
        print(f"[Calibration] angle(u1_cal, v1_cal) = {angle_after_1:.3f}° (target {angle1_final:.3f}°)")
        print(f"[Calibration] angle(u2_cal, v2_cal) = {angle_after_2:.3f}° (target {angle1_final:.3f}°)")
        print(f"[Twist] before calibration = {twist_angle_before_calibrate:.3f}°")
        print(f"[Twist] after  calibration = {twist_angle_after_calibrate:.3f}°")
        print(f"[Ratio] before calibration |u1|/|u2| = {ratio_before_1:.3f},  |v1|/|v2| = {ratio_before_2:.3f}")
        print(f"[Ratio] after calibration |u1|/|u2| = {ratio_after_1:.3f},  |v1|/|v2| = {ratio_after_2:.3f}")


        dict_1 = {
            "A": A,
            "u1_cal": u1_cal, "v1_cal": v1_cal,
            "u2_cal": u2_cal, "v2_cal": v2_cal,
            "u1_ref": u1_ref, "v1_ref": v1_ref,
            "angle_target": angle1_final,
            "angle_before": angle_vector1,
            "angle_after_1": angle_after_1,
            "angle_after_2": angle_after_2,
            "twist_angle_before": twist_angle_before_calibrate,
            "twist_angle_after": twist_angle_after_calibrate,
        }


        if return_info:
            return dict_1
