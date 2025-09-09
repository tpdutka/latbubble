#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code performs a 3D lattice simulation of a scalar field coupled to a thermal bath with a defined potential.
# During the simulation full lattice data is saved frequently (.h5) and the simulation is halted once a "well-past" super-critical
# bubble is detected. Afterwards the full lattice data per time step is cropped to a small sub-lattice around this bubble.
# Used to observe behaviour of field profile, phi(r), with time as it evolves from subcritical to supercritical.
# This repeats for given number of times

import math
import os
import numpy as np
import shutil
import traceback
import multiprocessing as mp
import h5py

# From package
from latticesim.core.integrators import (euler, rk2, rk4, leapfrog, forest_ruth)
from latticesim.core.laplacians import (laplacian_Oa4, laplacian_Oa2)
from latticesim.core.counterterms import get_counterterms_constants, get_counterterms
from latticesim.utils.io import extract_subgrid,radial_profile


# Lattice Simulation Params - Mostly Adjustable

# order - order of lattice counterterms, 2 = NNLO, 1 = NLO, 0 = LO, anything else = none
# improved - whether using Laplacian with O(dx^4) or O(dx^2) error, improved and unimproved respectively
# mom_refresh - whether using `partial momentum refreshment' in time evolution
order, improved, mom_refresh = 2, True, True

integrator_choice = 'forest_ruth'

Lx = Ly = Lz = 100
dx = dy = dz = 0.75
Nx = Ny = Nz = int(2*Lx/dx)
print("Lx, Nx, dx:", Lx, Nx, dx)

# T = total simulation time
T = 50.0
dt = 0.005 # probably require dt < sqrt(V''(phi_min))
num_steps = math.ceil(T / dt)

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

# subgrid, used for cropping
subgrid_size = 30

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

bubble_threshold = 0.9 * phi_glob
prebubble_threshold = 5

summary_file = "bubble_times_summary.txt"
summary_lock = mp.Lock()

laplacian = laplacian_Oa4(dx, dy, dz, C2, lam_lam, NNLO) if improved else laplacian_Oa2(dx, dy, dz, C2, lam_lam, NNLO)


stat_save_step = 2
data_save_step = 4
progress_save_step = 20


def run_simulation(run_num):

    seed_no = run_num + 11235 * 2
    np.random.seed(seed_no)    
    print("seed no: ", seed_no)   

    run_id = f"{run_num:03d}"
    run_dir = f"run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    temp_h5 = os.path.join(run_dir, f"temp_full_data_run_{run_id}.h5")
    h5_final = os.path.join(run_dir, f"subgrids_run_{run_id}.h5")

    frac_output = os.path.join(run_dir, f"global_res_{run_id}.txt")
    with open(frac_output, "w") as f_frac:
        f_frac.write("# t\t<phi^2>\t<phi>\t<|phi|>\tfrac_{|phi| < phi_barrier}\n")

    print(f"\n=== Starting run {run_id} ===")

    phi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    pi = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    bubble_found = False
    prebubble_found = False
    t_prebubble = None
    t_bubble = None

    with h5py.File(temp_h5, "w") as tempf:
        for step in range(num_steps):
            t = step * dt
            phi, pi = integrator(phi, pi, dt, dx, dy, dz, D, friction, dV_phi, laplacian, mom_refresh)

            two_pt_corr = np.mean(phi**2)
            one_pt_corr = np.mean(phi)
            abs_phi_mean = np.mean(np.abs(phi))
            frac_left_of_barrier = np.mean(np.abs(phi) < phi_barrier_1)

            if step % stat_save_step == 0:
                with open(frac_output, "a") as f_frac:
                    f_frac.write(f"{t:.5f}\t{two_pt_corr:.6e}\t{one_pt_corr:.6e}\t{abs_phi_mean:.6e}\t{frac_left_of_barrier:.6f}\n")

            if step % data_save_step == 0:
                g = tempf.create_group(f"{t:.5f}")
                g.create_dataset("phi", data=phi)
                g.create_dataset("pi", data=pi)

            if step % progress_save_step == 0:
                with open(os.path.join(run_dir, "progress.txt"), "w") as pf:
                    pf.write(f"Step: {step} / {num_steps}   (t = {t:.5f})\n")

            if not bubble_found and np.any(np.abs(phi) > bubble_threshold):
                bubble_coord = tuple(np.argwhere(np.abs(phi) > bubble_threshold)[0])
                print(f"Bubble detected at {bubble_coord} (t = {t:.5f})")
                t_bubble = t
                bubble_found = True
                break

            if not prebubble_found and np.any(np.abs(phi) > prebubble_threshold):
                t_prebubble = t
                prebubble_found = True

    if not bubble_found:
        print(f"No bubble detected in run {run_id}, skipping.")
        shutil.rmtree(run_dir)
        return

    with summary_lock:
        with open(summary_file, "a") as f:
            pre_t_str = f"{t_prebubble:.5f}" if t_prebubble is not None else "N/A"
            bub_t_str = f"{t_bubble:.5f}" if t_bubble is not None else "N/A"
            f.write(f"{run_id}\t{pre_t_str}\t{bub_t_str}\n")


    phi_r_file = os.path.join(run_dir, f"phi_r_vs_t_run_{run_id}.txt")
    abs_phi_r_file = os.path.join(run_dir, f"abs_phi_r_vs_t_run_{run_id}.txt")
    pi_r_file = os.path.join(run_dir, f"pi_r_vs_t_run_{run_id}.txt")
    abs_pi_r_file = os.path.join(run_dir, f"abs_pi_r_vs_t_run_{run_id}.txt")
    counts_file = os.path.join(run_dir, f"counts_vs_t_run_{run_id}.txt")


    try:
        with open(phi_r_file, "w") as fphi, open(abs_phi_r_file, "w") as fabsphi, \
             open(pi_r_file, "w") as fpi, open(abs_pi_r_file, "w") as fabspi, \
             open(counts_file, "w") as fcount, \
             h5py.File(h5_final, "w") as h5out, h5py.File(temp_h5, "r") as tempf:

            for t_key in sorted(tempf.keys(), key=lambda s: float(s)):
                t = float(t_key)
                phi = tempf[t_key]["phi"][:]
                pi = tempf[t_key]["pi"][:]

                phi_sub = extract_subgrid(phi, bubble_coord)
                pi_sub = extract_subgrid(pi, bubble_coord)

                if phi_sub.shape != (subgrid_size, subgrid_size, subgrid_size):
                    raise ValueError(f"Subgrid shape mismatch: got {phi_sub.shape} from coord {bubble_coord}")

                phi_r, abs_phi_r, counts = radial_profile(phi_sub)
                pi_r, abs_pi_r, _ = radial_profile(pi_sub)

                g = h5out.create_group(f"{t:.5f}")
                g.create_dataset("phi_subgrid", data=phi_sub)
                g.create_dataset("pi_subgrid", data=pi_sub)
                g.create_dataset("phi_r", data=phi_r)
                g.create_dataset("abs_phi_r", data=abs_phi_r)
                g.create_dataset("pi_r", data=pi_r)
                g.create_dataset("abs_pi_r", data=abs_pi_r)
                g.create_dataset("counts", data=counts)

                fphi.write(f"{t:.5f}\t" + "\t".join(map("{:.8f}".format, phi_r)) + "\n")
                fabsphi.write(f"{t:.5f}\t" + "\t".join(map("{:.8f}".format, abs_phi_r)) + "\n")
                fpi.write(f"{t:.5f}\t" + "\t".join(map("{:.8f}".format, pi_r)) + "\n")
                fabspi.write(f"{t:.5f}\t" + "\t".join(map("{:.8f}".format, abs_pi_r)) + "\n")
                fcount.write(f"{t:.5f}\t" + "\t".join(map(str, counts)) + "\n")

            r_bins_file = os.path.join(run_dir, f"r_bins_run_{run_id}.txt")
            with open(r_bins_file, "w") as fr:
                fr.write("# r_bin_centers (integer lattice radii)\n")
                r_int = np.arange(len(phi_r))
                fr.write(" ".join(str(int(r)) for r in r_int) + "\n")

        os.remove(temp_h5)

    except Exception as e:
        print(f"Run {run_id} failed at bubble processing: {e}")
        traceback.print_exc()

    finally:
        if os.path.exists(temp_h5):
            try: os.remove(temp_h5)
            except: pass


if __name__ == "__main__":

    if os.path.exists(summary_file):
        os.remove(summary_file)

    # compute time for first bubble, per run
    with open(summary_file, "w") as f:
        f.write("# run_id\tt_prebubble\tt_bubble\n")


    N_runs = 50
    broken_runs = []

    # Identify runs that are incomplete
    for run_id in range(1, N_runs + 1):
        run_dir = f"run_{run_id:03d}"
        temp_h5 = os.path.join(run_dir, f"temp_full_data_run_{run_id:03d}.h5")

        # Incomplete if temp file exists OR run dir missing expected final HDF5
        final_h5 = os.path.join(run_dir, f"subgrids_run_{run_id:03d}.h5")
        if os.path.exists(temp_h5) or not os.path.exists(final_h5):
            broken_runs.append(run_id)

    if broken_runs:
        print(f"Broken/incomplete runs detected: {broken_runs}")
        for run_id in broken_runs:
            run_dir = f"run_{run_id:03d}"
            print(f"Deleting {run_dir}...")
            shutil.rmtree(run_dir, ignore_errors=True)
        start_run = min(broken_runs)
    else:
        print("No broken runs detected. Starting from 1.")
        start_run = 1

    for run_id in range(start_run, N_runs + 1):
        if run_id in broken_runs or not os.path.exists(f"run_{run_id:03d}"):
            run_simulation(run_id)
        else:
            print(f"Skipping run_{run_id:03d} (already complete)")

    print("\nAll runs finished.")
