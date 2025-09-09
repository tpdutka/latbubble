#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code performs a 3D lattice simulation of a scalar field coupled to a thermal bath with a defined potential.
# During the simulation you are able to save various txt files of lattice statistical parameters, e.g. <phi^2>, <phi> etc,
# you can save histograms of phi (or pi) data as well as full lattice data.
# Finally, 2D plots can be made for defined z_slices of lattice data according to a plotting scheme of your choosing 

import math
import os
from collections import deque
import numpy as np

# From package
from latticesim.core.integrators import (euler, rk2, rk4, leapfrog, forest_ruth)
from latticesim.core.laplacians import (laplacian_Oa4, laplacian_Oa2)
from latticesim.core.counterterms import get_counterterms_constants, get_counterterms
from latticesim.utils.plotting import (save_slice_plot_old_linear, save_slice_plot_old_sqrt,
 save_slice_plot_old_log, save_slice_plot_new_linear, save_slice_plot_new_sqrt, save_slice_plot_new_log,)
from latticesim.utils.io import save_histogram

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
phi_barrier_2 = phi_barrier * temperature


friction = 1 
D = 2 * temperature * friction

# thresholds used for bubble detection/plotting if desired
# threshold_11 = temperature
threshold_12 = phi_barrier_1
threshold_2  = phi_glob

first_bubble_threshold = phi_glob / 2


# Counterterms & Laplacian
Sigma, x_i, C1, C2, C3 = get_counterterms_constants(improved)
LO, NLO, NNLO = get_counterterms(order)
dV_phi = potential_derivative(eps_eps_3, lam_lam_3, big_eps_3, Sigma, x_i, C1, C2, C3,
                              dx, LO, NLO, NNLO)

if improved:
    laplacian = laplacian_Oa4(dx, dy, dz, C2, lam_lam, NNLO)
else:
    laplacian = laplacian_Oa2(dx, dy, dz, C2, lam_lam, NNLO)


# If you want multiple plot schemes, otherwise use only one
plot_schemes = {
    "old_linear": save_slice_plot_old_linear,
    "old_sqrt":   save_slice_plot_old_sqrt,
    "old_log":    save_slice_plot_old_log,
    "new_linear": save_slice_plot_new_linear,
    "new_sqrt":   save_slice_plot_new_sqrt,
    "new_log":    save_slice_plot_new_log,
}

try:
    integrator = {
        'euler': euler, 'rk2': rk2, 'rk4': rk4, 'leapfrog': leapfrog, 'forest_ruth': forest_ruth
    }[integrator_choice]
except KeyError as e:
    raise ValueError(f"Unknown integrator '{integrator_choice}'") from e


# save intervals
time_interval_time    = 0.01 
time_interval_space   = 1.0  
time_interval_space_2 = 0.1 
time_interval_hist    = 0.1  

def to_steps(interval, dt):
    return max(1, int(interval / dt))

step_interval_time    = to_steps(time_interval_time, dt)
step_interval_space   = to_steps(time_interval_space, dt)
step_interval_space_2 = to_steps(time_interval_space_2, dt)
step_interval_hist    = to_steps(time_interval_hist, dt)


# Stability tracking
stability_buffer_length = 50
stability_tolerance = 1e-2
avg_abs_phi_buffer = deque(maxlen=stability_buffer_length)

num_runs = 5

for run in range(1, num_runs + 1):
    run_folder = f"run_{run}"
   
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(os.path.join(run_folder, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_folder, "results"), exist_ok=True)
    os.makedirs(os.path.join(run_folder, "plots"), exist_ok=True)
    
    hist_dir = os.path.join(run_folder, "results", "hist")
    os.makedirs(hist_dir, exist_ok=True)

    for scheme_name in plot_schemes:
        scheme_dir = os.path.join(run_folder, "plots", scheme_name)
        os.makedirs(scheme_dir, exist_ok=True)

    output_specs = {
        "1pt_corr_vs_time.txt":             "t\t<phi>\t<|phi|>\n",
        "2pt_corr_vs_time.txt":             "t\t<phi^2>\n",
        "frac_vs_time.txt":                 "t\tfrac_in_glob_min\n",
        "left_of_barrier_vs_time.txt":      "t\tfrac_absphi<barrier1\tfrac_absphi<barrier2\n",
        "frac_vs_time_wider.txt":           "t\tfrac_wide\tfrac_wider\tfrac_widerer\tfrac_narrow\n",
    }

    output_files = {}
    try:
        for fname, header in output_specs.items():
            path = os.path.join(run_folder, "results", fname)
            f = open(path, "w", buffering=1) 
            f.write(header)
            output_files[fname] = f
    except Exception:
        for f in output_files.values():
            try: f.close()
            except: pass
        raise


    # which z_slices to fix (for 2D plotting)
    z_slices = [0, Nz//4, Nz//2, 3*Nz//4]

    phi   = np.zeros((Nx, Ny, Nz))
    phi_t = np.zeros((Nx, Ny, Nz))

    count_1 = 0
    count_2 = 0
    first_bubble_found = 0


    for step in range(num_steps):

        if step % max(1, (num_steps // 100)) == 0:
            print(f"\rRun {run}, Progress: {100 * step // num_steps}%", end="")

        # Evolve
        phi, phi_t = integrator(phi, phi_t, dt, dx, dy, dz, D, friction, dV_phi, laplacian, mom_refresh)
        avg_abs_phi = np.mean(np.abs(phi))
        avg_abs_phi_buffer.append(avg_abs_phi)

        # Detect "first bubble" to add to z_slices (to plot a guaranteed bubble, and the first)
        if (np.max(np.abs(phi)) > first_bubble_threshold) and (first_bubble_found == 0):
            indices = np.where(np.abs(phi) > first_bubble_threshold)
            z_indices = indices[2]
            if len(z_indices) > 0:
                z_slices.append(z_indices[0])
            first_bubble_found = 1

        # store lattice statistical data
        if step % step_interval_time == 0:
            t_now = (step + 1) * dt

            # basic correlators
            two_pt_corr   = np.mean(phi**2)
            one_pt_corr_1 = np.mean(phi)

            # fractions near minima/barriers
            frac_at_min         = np.mean(np.abs(phi - phi_glob) <= 0.01 * phi_glob)
            frac_at_min_narrow  = np.mean(np.abs(phi - phi_glob) <= 0.001 * phi_glob)
            frac_at_min_wide    = np.mean(np.abs(phi - phi_glob) <= 0.05 * phi_glob)
            frac_at_min_wider   = np.mean(np.abs(phi - phi_glob) <= 0.1 * phi_glob)
            frac_at_min_widerer = np.mean(np.abs(phi - phi_glob) <= 0.5 * phi_glob)

            frac_left_of_barrier_1 = np.mean(np.abs(phi) < phi_barrier_1)
            frac_left_of_barrier_2 = np.mean(np.abs(phi) < phi_barrier_2)

            # write once per file (re-use the handles from output_files)
            output_files["1pt_corr_vs_time.txt"]       .write(f"{t_now:.5f}\t{one_pt_corr_1:.5f}\t{avg_abs_phi:.5f}\n")
            output_files["2pt_corr_vs_time.txt"]       .write(f"{t_now:.5f}\t{two_pt_corr:.5f}\n")
            output_files["frac_vs_time.txt"]           .write(f"{t_now:.5f}\t{frac_at_min:.5f}\n")
            output_files["left_of_barrier_vs_time.txt"].write(f"{t_now:.5f}\t{frac_left_of_barrier_1:.5f}\t{frac_left_of_barrier_2:.5f}\n")
            output_files["frac_vs_time_wider.txt"]     .write(
                f"{t_now:.5f}\t{frac_at_min_wide:.5f}\t{frac_at_min_wider:.5f}\t"
                f"{frac_at_min_widerer:.5f}\t{frac_at_min_narrow:.5f}\n"
            )

        # store full lattice data
        if step % step_interval_space == 0:
            filename_phi = os.path.join(run_folder, "data", f"phi_data_{count_1}.npy")
            np.save(filename_phi, phi)
            count_1 += 1

        # plot 2D lattice slices
        if step % step_interval_space_2 == 0:
            for index, z_slice_index in enumerate(z_slices):
                for scheme_name, plot_func in plot_schemes.items():
                    scheme_dir = os.path.join(run_folder, "plots", scheme_name)
                    slice_subdir = os.path.join(scheme_dir, f"slice_{index+1}")
                    os.makedirs(slice_subdir, exist_ok=True)

                    plot_func(
                        phi, dt,
                        Lx, Ly, Nx, Ny,
                        step, z_slice_index,
                        threshold_12, threshold_2,
                        count_2,
                        output_dir=slice_subdir
                    )
            count_2 += 1

        # Histogram of phi data
        if step % step_interval_hist == 0:
            # Freedman–Diaconis binning => 'fd' so bin sizes adapt dynamically
            phi_min, phi_max = phi.min(), phi.max()
            absphi_max = np.abs(phi).max()

            # Histogram for phi
            hist_phi, edges_phi = np.histogram(phi, bins="fd", range=(phi_min, phi_max))
            centers_phi = 0.5 * (edges_phi[:-1] + edges_phi[1:])

            # Histogram for |phi|
            hist_abs, edges_abs = np.histogram(np.abs(phi), bins="fd", range=(0, absphi_max))
            centers_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])

            # Save them in the dedicated 'hist' folder
            save_histogram(os.path.join(hist_dir, f"hist_phi_step_{step}.txt"),
                           centers_phi, hist_phi, title="phi")
            save_histogram(os.path.join(hist_dir, f"hist_abs_phi_step_{step}.txt"),
                           centers_abs, hist_abs, title="|phi|")

        # stop sim. early if reached minimum (phi data doesn't change)
        if len(avg_abs_phi_buffer) == stability_buffer_length:
            max_avg_abs_phi = max(avg_abs_phi_buffer)
            min_avg_abs_phi = min(avg_abs_phi_buffer)
            delta_avg_abs_phi = max_avg_abs_phi - min_avg_abs_phi

            if (delta_avg_abs_phi < stability_tolerance) and (min_avg_abs_phi > 0.9*phi_glob):
                print(f"\nSystem reached stability in run {run} at step {step}")
                break

    print("\nSimulation complete.")
    for f in output_files.values():
        try:
            f.close()
        except:
            pass