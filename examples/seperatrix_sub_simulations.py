#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code performs a 3D lattice simulation of a scalar field coupled to a thermal bath with a defined potential.
# This example specifically requires input data (in the .h5 format) from which the simulation can draw initial conditions for phi and pi.
# Using these initial conditions, N_CLONES sub simulations are performed starting from this initial condition and the trajectory of 
# <phi^2>, etc are tracked for some small time period. One output is generated per sub simulation which tracks the time evolution.

# A pre-thermalisation stage is generated before the desired initial condition is inserted into the simulation and secdondary, "B" simulation,
# is also performed without this initial condition being inserted to compare the impact that the initial profile has on the trajectories.
# Crucially the two simulations (with the initial profile and B) share common random numbers (CNR) such that the initial profile is key 
# difference between the two. 

# This code assumes data.h5 which is generated in the format of examples/simulation_with_bubble_data_saving.py, e.g. phi,pi data centered around
# a bubble at various time steps. Without such specific data code will not work out of the box.

# The code will perform a fixed number of subsimulation per time step, with an adjustsable number of time differnece between each snapshot.
# The code can automatically define an initial time from which to simulate the data based on the inputed time step size (the time step between initial data)
# not the time step of the integrator) as well as the largest time step recorded in the initial data.h5 file given.

# code can be run in two ways:
# 1) Bulk mode over a directory (for me labeled like run_xxx containing the h5 data):  python code_name.py /path/to/parent_dir
# 2) Single snapshot: python code_name.py subgrids_run_xxx.h5 TIME_KEY_FLOAT



import math
import os
import numpy as np
import shutil
import traceback
import h5py
import sys
import re
import glob


# From package
from latticesim.core.integrators import (euler, rk2, rk4, leapfrog, forest_ruth)
from latticesim.core.laplacians import (laplacian_Oa4, laplacian_Oa2)
from latticesim.core.counterterms import get_counterterms_constants, get_counterterms
from latticesim.utils.io import ensure_dir, load_subgrid, list_time_keys, nearest_down_to_grid


# Lattice Simulation Params - Mostly Adjustable

# order - order of lattice counterterms, 2 = NNLO, 1 = NLO, 0 = LO, anything else = none
# improved - whether using Laplacian with O(dx^4) or O(dx^2) error, improved and unimproved respectively
# mom_refresh - whether using `partial momentum refreshment' in time evolution
order, improved, mom_refresh = 2, True, True

integrator_choice = 'forest_ruth'

dx = dy = dz = 0.75
dt = 0.005
T_TOTAL = 5.0           # duration AFTER insertion (the “real” run)
PRE_TIME = 2.0          # evolve zero-initialized bath for this long, then insert subgrid
N_CLONES = 100
SEED_BASE = 123

TIME_STEP = 0.5         # time step between each initial condition snapshot
MAX_TIME_SNAPSHOTS = 6  # try up to this many snapshots per run (e.g., t_i, t_(i-0.5), ..., 6 values)

# box sizes
N_TARGET = 40  

RECORD_EVERY = 40 

# seam & halo (environment control)
SEAM_THICK = 2          # cells to feather at the core’s edge upon insertion (0 disables)
HALO_THICK = 3          # outer boundary layers to stabilize (0 disables)
HALO_MODE = "fixed"     # "fixed" to clamp halo to FV=0 each step, or None

# spheres around the centre to monitor (very close to the initial bubble wall)
SPHERE_RADII = [3, 5, 7]   
SPHERE_UNITS = "cells"     # "cells" (radii in lattice cells) or "phys" (radii in physical units)

# Potential - This case specifically for: dV = eps_eps * phi + (1/3!) lam_lam * phi**3  + (1/5!) big_eps * phi**5
temperature = 1
lam_lam     = -2
big_eps     = 1e-2
phi_barrier = 0.3

# dimensionally reduced parameters
eps_eps = (np.abs(lam_lam) / 12) * phi_barrier**2 * temperature**2
lam_lam_3 = lam_lam * temperature
eps_eps_3 = eps_eps
big_eps_3 = big_eps / temperature**2

print("Signs:", np.sign(eps_eps_3), np.sign(lam_lam_3), np.sign(big_eps_3))

# lattice counter terms at different orders can be turned off using LO/NLO/NNLO -> 0
def potential_derivative(mass, lam_lam, eps, Sigma, x_i, C1, C2, C3, dx, LO, NLO, NNLO):

    lam2_dx2_over_16pi2 = (lam_lam**2 * dx**2) / (16 * np.pi**2)
    log_term = np.log(6 / (dx * np.abs(lam_lam))) + C3 - Sigma * x_i

    Z_phi = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_m   = 1 + NLO * ((x_i * lam_lam * dx) / (8 * np.pi)) + NNLO * lam2_dx2_over_16pi2 * (x_i**2/4 - C1/2 - C2/6)

    del_m   = LO * (-(Sigma * lam_lam) / (8 * np.pi * dx) + lam2_dx2_over_16pi2 * log_term)
    del_lam = LO * (-(eps * Sigma) / (8*np.pi*dx)) + NLO * ((3 * x_i * lam_lam**2 * dx) / (8*np.pi)) \
              + NNLO * ((lam_lam**2 * dx**2) / (64*np.pi**3)) * ((3/4)*x_i**2 - 3*C1 - C2/3)
    del_eps = LO * ((5 * eps**2) / (3 * 16 * np.pi**2)) * log_term

    a1 = Z_phi * (mass + del_m)
    a3 = (1/6)   * Z_phi**2 * (lam_lam + del_lam)
    a5 = (1/120) * Z_phi**3 * (eps + del_eps)

    def dV(phi):
        return a1 * phi + a3 * phi**3 + a5 * phi**5
    return dV


phi_glob      = 2 * np.sqrt(5) * np.sqrt(np.abs(lam_lam) / big_eps)
phi_barrier_1 = phi_barrier * temperature / np.sqrt(2)

friction = 1 
D = 2 * temperature * friction

# Counterterms & Laplacian
Sigma, x_i, C1, C2, C3 = get_counterterms_constants(improved)
LO, NLO, NNLO = get_counterterms(order)
dV_phi = potential_derivative(eps_eps_3, lam_lam_3, big_eps_3, Sigma, x_i, C1, C2, C3,
                              dx, LO, NLO, NNLO)

try:
    integrator = {
        'euler': euler, 'rk2': rk2, 'rk4': rk4, 'leapfrog': leapfrog, 'forest_ruth': forest_ruth
    }[integrator_choice]
except KeyError as e:
    raise ValueError(f"Unknown integrator '{integrator_choice}'") from e

# can also pick Neumann or dirichlet but below a thickish halo is applied on the lattice edge per time step so choices don't matter that much
laplacian = laplacian_Oa4(dx, dy, dz, C2, lam_lam, NNLO) if improved else laplacian_Oa2(dx, dy, dz, C2, lam_lam, NNLO)


# specific functions to help 'seam' the data to the prethermalised bath

def center_insert(core, host, seam_thick=0, blend_to="bath"):
    Nc = core.shape[0]; Nh = host.shape[0]
    assert Nc <= Nh and core.shape == (Nc, Nc, Nc) and host.shape == (Nh, Nh, Nh)
    s = (Nh - Nc)//2; e = s + Nc

    if seam_thick > 0 and blend_to == "bath":
        bath = host[s:e, s:e, s:e].copy()

    host[s:e, s:e, s:e] = core

    if seam_thick <= 0:
        return host

    idx = np.indices((Nc, Nc, Nc))
    d_face = np.minimum.reduce([
        idx[0], Nc - 1 - idx[0],
        idx[1], Nc - 1 - idx[1],
        idx[2], Nc - 1 - idx[2]
    ])
    shell = (d_face < seam_thick)
    alpha = np.clip((seam_thick - d_face) / float(seam_thick), 0.0, 1.0)
    patch = host[s:e, s:e, s:e]

    if blend_to == "zero":
        target = 0.0
        patch[shell] = (1.0 - alpha[shell]) * patch[shell] + alpha[shell] * target
    elif blend_to == "bath":
        patch[shell] = (1.0 - alpha[shell]) * patch[shell] + alpha[shell] * bath[shell]
    else:
        raise ValueError("blend_to must be 'bath' or 'zero'")

    host[s:e, s:e, s:e] = patch
    return host

def apply_halo(phi, pi, halo_thick, mode="fixed"):
    if halo_thick <= 0 or mode is None:
        return
    N = phi.shape[0]
    idxs = list(range(halo_thick)) + list(range(N - halo_thick, N))
    phi[idxs, :, :] = 0.0;  pi[idxs, :, :] = 0.0
    phi[:, idxs, :] = 0.0;  pi[:, idxs, :] = 0.0
    phi[:, :, idxs] = 0.0;  pi[:, :, idxs] = 0.0

def build_centered_sphere_masks(N, dx, radii, units="cells"):
    c = N // 2
    x = np.arange(N) - c
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X*X + Y*Y + Z*Z)
    if units == "phys":
        r = r * dx
    masks, labels = [], []
    for R in radii:
        masks.append(r < (R + 1e-12))
        suffix = f"r<{R}" + ("" if units == "phys" else "cells")
        labels.append(suffix)
    return masks, labels

def _has_data_rows(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                return True
        return False
    except Exception:
        return False

def is_run_complete(out_dir, run_id, require_spheres=False):
    tag = f"run_{run_id:03d}"
    moments_A = os.path.join(out_dir, f"{tag}_moments.txt")
    moments_B = os.path.join(out_dir, f"{tag}_moments_B.txt")

    if not (_has_data_rows(moments_A) and _has_data_rows(moments_B)):
        return False

    if require_spheres:
        spheres_A = os.path.join(out_dir, f"{tag}_moments_spheres.txt")
        spheres_B = os.path.join(out_dir, f"{tag}_moments_spheres_B.txt")
        if not (_has_data_rows(spheres_A) and _has_data_rows(spheres_B)):
            return False

    return True

def cleanup_run_files(out_dir, run_id):
    tag = f"run_{run_id:03d}_"
    for p in glob.glob(os.path.join(out_dir, tag + "*.txt")):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

def first_incomplete_run_id(out_dir, n_clones, require_spheres=False):
    for rid in range(n_clones):
        if not is_run_complete(out_dir, rid, require_spheres=require_spheres):
            return rid
    return None



# ---------- one paired clone (A: with core, B: without) ----------
def run_clone(core_phi, core_pi, steps_after, pre_steps, seed, out_dir, run_id):
    # PRE-THERMALIZE BATH ONCE
    np.random.seed(seed)
    Nc = core_phi.shape[0]
    N  = N_TARGET if N_TARGET is not None else Nc
    assert N >= Nc, "N_TARGET must be >= core subgrid size"

    phi_bath = np.zeros((N, N, N), dtype=np.float64)
    pi_bath  = np.zeros((N, N, N), dtype=np.float64)

    # prethermalisation
    t = 0.0
    for _ in range(pre_steps):
        apply_halo(phi_bath, pi_bath, HALO_THICK, HALO_MODE)
        phi_bath, pi_bath = integrator(phi_bath, pi_bath, dt, dx, dy, dz, D, friction, dV_phi, laplacian, mom_refresh)
        t += dt

    phiA = phi_bath.copy(); piA = pi_bath.copy()   # with core
    phiB = phi_bath.copy(); piB = pi_bath.copy()   # baseline

    center_insert(core_phi, phiA, seam_thick=SEAM_THICK, blend_to="bath")
    center_insert(core_pi,  piA,  seam_thick=SEAM_THICK, blend_to="bath")

    # Reset time origin AFTER insertion
    t = 0.0

    sphere_masks, sphere_labels = build_centered_sphere_masks(N, dx, SPHERE_RADII, SPHERE_UNITS)

    moments_path_A = os.path.join(out_dir, f"run_{run_id:03d}_moments.txt")
    moments_path_B = os.path.join(out_dir, f"run_{run_id:03d}_moments_B.txt")
    spheres_path_A = os.path.join(out_dir, f"run_{run_id:03d}_moments_spheres.txt")
    spheres_path_B = os.path.join(out_dir, f"run_{run_id:03d}_moments_spheres_B.txt")

    with open(moments_path_A, "w") as fmA, open(moments_path_B, "w") as fmB, \
         open(spheres_path_A, "w") as fsA, open(spheres_path_B, "w") as fsB:

        fmA.write("# with insert\n# t <phi^2> <phi> <|phi|> <pi^2> <pi> <|pi|>\n")
        fmB.write("# baseline (no insert)\n# t <phi^2> <phi> <|phi|> <pi^2> <pi> <|pi|>\n")

        header = ["# t"]
        for lab in sphere_labels:
            header += [f"<phi^2>_{lab}", f"<phi>_{lab}", f"<|phi|>_{lab}",
                       f"<pi^2>_{lab}", f"<pi>_{lab}", f"<|pi|>_{lab}", f"Npts_{lab}"]
        fsA.write(" ".join(header) + "\n")
        fsB.write(" ".join(header) + "\n")

        # EVOLVE WITH COMMON RANDOM NUMBERS
        steps_after = int(steps_after)
        for step in range(steps_after):
            if (step % RECORD_EVERY) == 0:
                # A
                fmA.write("{:.6f} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e}\n".format(
                    t, np.mean(phiA*phiA), np.mean(phiA), np.mean(np.abs(phiA)),
                    np.mean(piA*piA),  np.mean(piA),  np.mean(np.abs(piA))
                ))
                rowA = [f"{t:.6f}"]
                for mask in sphere_masks:
                    p = phiA[mask]; q = piA[mask]; n = p.size
                    if n == 0:
                        rowA += ["nan"]*6 + ["0"]
                    else:
                        rowA += [f"{np.mean(p*p):.8e}", f"{np.mean(p):.8e}", f"{np.mean(np.abs(p)):.8e}",
                                 f"{np.mean(q*q):.8e}", f"{np.mean(q):.8e}", f"{np.mean(np.abs(q)):.8e}", f"{n}"]
                fsA.write(" ".join(rowA) + "\n")

                # B
                fmB.write("{:.6f} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e}\n".format(
                    t, np.mean(phiB*phiB), np.mean(phiB), np.mean(np.abs(phiB)),
                    np.mean(piB*piB),  np.mean(piB),  np.mean(np.abs(piB))
                ))
                rowB = [f"{t:.6f}"]
                for mask in sphere_masks:
                    p = phiB[mask]; q = piB[mask]; n = p.size
                    if n == 0:
                        rowB += ["nan"]*6 + ["0"]
                    else:
                        rowB += [f"{np.mean(p*p):.8e}", f"{np.mean(p):.8e}", f"{np.mean(np.abs(p)):.8e}",
                                 f"{np.mean(q*q):.8e}", f"{np.mean(q):.8e}", f"{np.mean(np.abs(q)):.8e}", f"{n}"]
                fsB.write(" ".join(rowB) + "\n")

            apply_halo(phiA, piA, HALO_THICK, HALO_MODE)
            apply_halo(phiB, piB, HALO_THICK, HALO_MODE)

            # COMMON RANDOM NUMBERS per step
            seed_step = (seed * 1000003 + step) % (2**32)

            np.random.seed(seed_step);  phiA, piA = integrator(phiA, piA, dt, dx, dy, dz, D, friction, dV_phi, laplacian, mom_refresh)
            np.random.seed(seed_step);  phiB, piB = integrator(phiB, piB, dt, dx, dy, dz, D, friction, dV_phi, laplacian, mom_refresh)

            t += dt

def process_run_folder(run_dir, max_time_snaps=MAX_TIME_SNAPSHOTS, step=TIME_STEP):
    # find the subgrids h5 (mother snapshot file)
    cand = sorted(glob.glob(os.path.join(run_dir, "subgrids_run_*.h5")))
    if not cand:
        print(f"[skip] No subgrids HDF5 in {run_dir}")
        return
    h5_path = cand[0]

    # list available times and choose starting t0
    times = list_time_keys(h5_path)
    if not times:
        print(f"[skip] No time groups in {h5_path}")
        return
    tf = times[-1]
    t0 = nearest_down_to_grid(tf, step)

    # build descending candidate times
    targets = []
    for k in range(max_time_snaps):
        tk = t0 - k*step
        gname = f"{tk:.5f}"
        with h5py.File(h5_path, "r") as f:
            if gname in f:
                targets.append(tk)
            else:
                continue
    if not targets:
        print(f"[skip] No matching snapshot times on {step}-grid in {h5_path}")
        return

    print(f"[{run_dir}] tf={tf:.5f} ⇒ start at t0={t0:.5f}, snapshots={len(targets)}: {', '.join(f'{t:.2f}' for t in targets)}")

    pre_steps   = int(round(PRE_TIME / dt))
    steps_after = int(round(T_TOTAL / dt))

    core_phi0, core_pi0 = load_subgrid(h5_path, targets[0])
    Nc = core_phi0.shape[0]
    N  = N_TARGET if N_TARGET is not None else Nc
    if N < Nc:
        raise ValueError(f"N_TARGET={N} must be >= core size Nc={Nc}")

    for t_key in targets:
        out_dir = ensure_dir(os.path.join(run_dir, f"paired_clones_t{t_key:.5f}_N{N}_PRE{PRE_TIME}"))

        rid0 = first_incomplete_run_id(out_dir, N_CLONES, require_spheres=False) 
        if rid0 is None:
            print(f"  [t={t_key:.5f}] already complete (N={N_CLONES}), skipping.")
            continue

        cleanup_run_files(out_dir, rid0)
        print(f"  [t={t_key:.5f}] resuming from run_{rid0:03d} (will recompute this id)")

        core_phi, core_pi = load_subgrid(h5_path, t_key)

        base = os.path.basename(os.path.abspath(run_dir))
        m = re.match(r"run_(\d+)", base)
        folder_id = int(m.group(1)) if m else 0

        for k in range(rid0, N_CLONES):
            seed = SEED_BASE + k + 10000019 * folder_id
            if is_run_complete(out_dir, k, require_spheres=False):
                continue
            cleanup_run_files(out_dir, k)
            run_clone(core_phi, core_pi, steps_after, pre_steps, seed, out_dir, k)

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage:")
        print("  1) Bulk mode over a directory of run_0xx folders:")
        print("       python code_name.py /path/to/parent_dir")
        print("  2) Single snapshot:")
        print("       python code_name.py subgrids_run_xxx.h5 TIME_KEY_FLOAT")
        sys.exit(1)

    # Bulk mode: argument is a directory containing run_0xx
    if len(args) == 1 and os.path.isdir(args[0]):
        target = os.path.abspath(args[0])

        if glob.glob(os.path.join(target, "subgrids_run_*.h5")):
            process_run_folder(target, MAX_TIME_SNAPSHOTS, TIME_STEP)
            print("Finished:", target)
            return

        run_dirs = sorted(d for d in glob.glob(os.path.join(target, "run_*")) if os.path.isdir(d))
        if not run_dirs:
            print(f"No run_* subdirectories found in {target}")
            sys.exit(0)

        for rd in run_dirs:
            process_run_folder(rd, MAX_TIME_SNAPSHOTS, TIME_STEP)
        print("All requested runs processed.")
        return

    # single-snapshot mode
    if len(args) == 2:
        h5_path = args[0]
        t_key   = float(args[1])

        core_phi, core_pi = load_subgrid(h5_path, t_key)
        Nc = core_phi.shape[0]
        N = N_TARGET if N_TARGET is not None else Nc
        if N < Nc:
            raise ValueError(f"N_TARGET={N} must be >= core size Nc={Nc}")

        pre_steps   = int(round(PRE_TIME / dt))
        steps_after = int(round(T_TOTAL / dt))

        base_dir = os.path.dirname(os.path.abspath(h5_path))
        out_dir = ensure_dir(os.path.join(base_dir, f"paired_clones_t{t_key:.5f}_N{N}_PRE{PRE_TIME}"))

        rid0 = first_incomplete_run_id(out_dir, N_CLONES, require_spheres=False)
        if rid0 is None:
            print(f"[single t={t_key:.5f}] already complete (N={N_CLONES}), nothing to do.")
            return

        cleanup_run_files(out_dir, rid0)
        print(f"[single t={t_key:.5f}] resuming from run_{rid0:03d}")

        base = os.path.basename(base_dir)
        m = re.match(r"run_(\d+)", base)
        folder_id = int(m.group(1)) if m else 0

        for k in range(rid0, N_CLONES):
            seed = SEED_BASE + k + 10000019 * folder_id 
            if is_run_complete(out_dir, k, require_spheres=False):
                continue
            cleanup_run_files(out_dir, k)
            run_clone(core_phi, core_pi, steps_after, pre_steps, seed, out_dir, k)

        print(f"Done single snapshot t={t_key:.5f}.")
        return

    print("Unrecognized arguments. See usage above.")
    sys.exit(1)

if __name__ == "__main__":
    main()
