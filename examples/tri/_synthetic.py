"""Small deterministic diffraction image for the independent tri examples."""
import numpy as np


def three_lattice_image():
    shape = (256, 256)
    origin = np.array([128., 128.])
    rr, cc = np.indices(shape)
    image = np.full(shape, 10., dtype=np.float32)
    bases = []
    for angle, amplitude in zip((0., 20., 40.), (9000., 7000., 5000.)):
        t = np.deg2rad(angle)
        rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        u, v = rotation @ np.array([0., 65.]), rotation @ np.array([65., 0.])
        bases.append((u, v))
        for a in range(-2, 3):
            for b in range(-2, 3):
                if (a == 0 and b == 0) or np.hypot(a, b) > 1.5:
                    continue
                point = origin + a * u + b * v
                image += amplitude * np.exp(-((rr-point[0])**2 + (cc-point[1])**2) / (2 * 1.5**2))
    image += 20000. * np.exp(-((rr-origin[0])**2 + (cc-origin[1])**2) / (2 * 2.**2))
    return image, origin, bases
