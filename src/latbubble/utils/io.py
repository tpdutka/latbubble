# src/latticesim/utils/io.py
from pathlib import Path
import h5py
import math
import os

def save_histogram(path: str | Path, bin_centers, counts, title: str | None = None):
    path = Path(path)
    with path.open("w") as f:
        f.write("# Freedman–Diaconis bins\n")
        f.write("# bin_center count\n")
        if title:
            f.write(f"# {title}\n")
        for c, n in zip(bin_centers, counts):
            f.write(f"{c:.6f} {n}\n")

def extract_subgrid(field, center):
    x0, y0, z0 = center
    N = field.shape[0]  # assumes cubic lattice (Nx = Ny = Nz)

    # Create index ranges with periodic wrapping
    x_idx = np.arange(x0 - half_N, x0 + half_N) % N
    y_idx = np.arange(y0 - half_N, y0 + half_N) % N
    z_idx = np.arange(z0 - half_N, z0 + half_N) % N

    return field[np.ix_(x_idx, y_idx, z_idx)]

def radial_profile(subgrid):
    N = subgrid.shape[0]
    center = (N // 2, N // 2, N // 2)
    x = np.arange(N) - center[0]
    y = np.arange(N) - center[1]
    z = np.arange(N) - center[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2).flatten()
    r_int = np.round(r).astype(int)

    flat = subgrid.flatten()
    abs_flat = np.abs(flat)
    max_r = r_int.max()

    vals, abs_vals, counts = [], [], []
    for radius in range(max_r + 1):
        mask = (r_int == radius)
        if np.any(mask):
            vals.append(flat[mask].mean())
            abs_vals.append(abs_flat[mask].mean())
            counts.append(mask.sum())
    return np.array(vals), np.array(abs_vals), np.array(counts)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_subgrid(h5_path, t_key_float):
    with h5py.File(h5_path, "r") as f:
        gname = f"{t_key_float:.5f}"
        if gname not in f:
            raise KeyError(f"time key {gname} not found in {h5_path}")
        g = f[gname]
        return g["phi_subgrid"][:], g["pi_subgrid"][:]

def list_time_keys(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = [float(k) for k in f.keys()]
    keys.sort()
    return keys

def nearest_down_to_grid(x, step):
    return math.floor(x/step)*step