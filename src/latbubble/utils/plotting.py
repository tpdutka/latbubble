# src/latticesim/utils/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt

def interpolate_color(c1, c2, alpha):
    return tuple((1 - alpha) * np.array(c1) + alpha * np.array(c2))

black       = (0.0, 0.0, 0.0) 
final_red   = (1.0, 0.3, 0.3) 
final_blue  = (0.3, 0.3, 1.0) 
mid_red     = (0.6, 0.3, 0.3) 
mid_blue    = (0.3, 0.3, 0.6) 

final_red_2 = (1, 0.5, 0.5)
final_blue_2 = (0.5, 0.5, 1)

final_red_hybrid  = interpolate_color(black, final_red_2, 1.0)
final_blue_hybrid = interpolate_color(black, final_blue_2, 1.0)


# messy examples of differnet plotting versions, e.g. change of colour and interpolations, best to make your own

def save_slice_plot(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = np.abs(phi[:, :, z_slice])
    colors = np.full(phi_slice.shape + (3,), 1.0)
    colors[phi_slice >= threshold_2] = (0, 0, 0)
    within = (phi_slice >= threshold_1) & (phi_slice < threshold_2)
    g = 1 - (phi_slice[within] - threshold_1) / (threshold_2 - threshold_1)
    colors[within] = np.stack([g, g, g], axis=-1)
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, color=Cf, marker='o', edgecolors='none')
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(f"{output_dir}/slice_plot_{count}.png"); plt.close()


def save_slice_plot_col(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.full(phi_slice.shape + (3,), 1.0)
    colors[np.abs(phi_slice) >= threshold_2] = (0, 0, 0)
    within = (np.abs(phi_slice) >= threshold_1) & (np.abs(phi_slice) < threshold_2)
    pos = within & (phi_slice > 0)
    neg = within & (phi_slice < 0)
    gp = 1 - (phi_slice[pos] - threshold_1) / (threshold_2 - threshold_1)
    gn = 1 - (np.abs(phi_slice[neg]) - threshold_1) / (threshold_2 - threshold_1)
    colors[pos] = np.stack([gp, np.zeros_like(gp), np.zeros_like(gp)], axis=-1)
    colors[neg] = np.stack([np.zeros_like(gn), np.zeros_like(gn), gn], axis=-1)
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, color=Cf, marker='o', edgecolors='none')
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(f"{output_dir}/slice_plot_{count}.png"); plt.close()


def save_slice_plot_col_2(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    final_color_positive = (1, 0.5, 0.5)
    final_color_negative = (0.5, 0.5, 1)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.ones(phi_slice.shape + (3,))
    within_gray = (np.abs(phi_slice) <= threshold_1)
    g = 1 - (np.abs(phi_slice[within_gray]) / threshold_1)
    colors[within_gray] = np.stack([g, g, g], axis=-1)
    within_grad = (np.abs(phi_slice) > threshold_1) & (np.abs(phi_slice) <= threshold_2)
    pos = within_grad & (phi_slice > 0)
    neg = within_grad & (phi_slice < 0)
    ip = ((phi_slice[pos] - threshold_1) / (threshold_2 - threshold_1)) ** 2
    ineg = ((np.abs(phi_slice[neg]) - threshold_1) / (threshold_2 - threshold_1)) ** 2
    colors[pos] = np.stack([ip * final_color_positive[0], ip * final_color_positive[1], ip * final_color_positive[2]], axis=-1)
    colors[neg] = np.stack([ineg * final_color_negative[0], ineg * final_color_negative[1], ineg * final_color_negative[2]], axis=-1)
    above = np.abs(phi_slice) > threshold_2
    colors[above & (phi_slice > 0)] = final_color_positive
    colors[above & (phi_slice < 0)] = final_color_negative
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=10)
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(f"{output_dir}/slice_plot_{count}.png"); plt.close()


def save_slice_plot_asymmetric(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, phi_meta, barrier_threshold, phi_global, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    final_color = (1, 0.5, 0.5)
    barrier_color = (0, 0, 0)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.ones(phi_slice.shape + (3,))
    meta = phi_slice <= phi_meta
    colors[meta] = (1, 1, 1)
    barrier = (phi_slice > phi_meta) & (phi_slice <= barrier_threshold)
    b = 1 - (phi_slice[barrier] - phi_meta) / (barrier_threshold - phi_meta)
    colors[barrier] = np.stack([b, b, b], axis=-1)
    to_global = (phi_slice > barrier_threshold) & (phi_slice <= phi_global)
    ig = ((phi_slice[to_global] - barrier_threshold) / (phi_global - barrier_threshold)) ** 2
    colors[to_global] = np.stack([
        ig * final_color[0] + (1 - ig) * barrier_color[0],
        ig * final_color[1] + (1 - ig) * barrier_color[1],
        ig * final_color[2] + (1 - ig) * barrier_color[2]
    ], axis=-1)
    near_global = phi_slice > phi_global
    colors[near_global] = final_color
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=10)
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(f"{output_dir}/slice_plot_{count}.png"); plt.close()


def save_slice_plot_old_linear(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.ones(phi_slice.shape + (3,))
    abs_phi = np.abs(phi_slice)
    small = abs_phi <= threshold_1
    a_small = abs_phi[small] / threshold_1
    colors[small] = np.array([interpolate_color((1,1,1), black, a) for a in a_small]).reshape(-1, 3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = (abs_phi[med] - threshold_1) / (threshold_2 - threshold_1)
    pos_med = med & (phi_slice > 0)
    neg_med = med & (phi_slice < 0)
    pos_idx = phi_slice[med] > 0
    neg_idx = phi_slice[med] < 0
    colors[pos_med] = np.array([interpolate_color(black, final_red, a) for a in a_med[pos_idx]]).reshape(-1, 3)
    colors[neg_med] = np.array([interpolate_color(black, final_blue, a) for a in a_med[neg_idx]]).reshape(-1, 3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red
    colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5)
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()


def save_slice_plot_old_sqrt(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.ones(phi_slice.shape + (3,))
    abs_phi = np.abs(phi_slice)
    small = abs_phi <= threshold_1
    a_small = np.sqrt(abs_phi[small] / threshold_1)
    colors[small] = np.array([interpolate_color((1,1,1), black, a) for a in a_small]).reshape(-1, 3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = np.sqrt((abs_phi[med] - threshold_1) / (threshold_2 - threshold_1))
    pos_med = med & (phi_slice > 0)
    neg_med = med & (phi_slice < 0)
    pos_idx = phi_slice[med] > 0
    neg_idx = phi_slice[med] < 0
    colors[pos_med] = np.array([interpolate_color(black, final_red, a) for a in a_med[pos_idx]]).reshape(-1, 3)
    colors[neg_med] = np.array([interpolate_color(black, final_blue, a) for a in a_med[neg_idx]]).reshape(-1, 3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red
    colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5)
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()


def save_slice_plot_old_log(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots", log_scale=10.0):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"t = {step * dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx)
    y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]
    colors = np.ones(phi_slice.shape + (3,))
    abs_phi = np.abs(phi_slice)
    tiny = threshold_1 / 100.0
    small = (abs_phi >= tiny) & (abs_phi <= threshold_1)
    val_small = abs_phi[small]; val_range = threshold_1 - tiny
    def log_map(v): return np.log(1 + log_scale*(v - tiny)) / np.log(1 + log_scale*val_range)
    a_small = [log_map(v) for v in val_small]
    colors[small] = np.array([interpolate_color((1,1,1), black, a) for a in a_small]).reshape(-1, 3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = (abs_phi[med] - threshold_1) / (threshold_2 - threshold_1)
    pos_med = med & (phi_slice > 0); neg_med = med & (phi_slice < 0)
    pos_idx = phi_slice[med] > 0; neg_idx = phi_slice[med] < 0
    colors[pos_med] = np.array([interpolate_color(black, final_red, a) for a in a_med[pos_idx]]).reshape(-1, 3)
    colors[neg_med] = np.array([interpolate_color(black, final_blue, a) for a in a_med[neg_idx]]).reshape(-1, 3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red
    colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5)
    plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()


def save_slice_plot_new_linear(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6,6)); plt.title(f"t = {step*dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx); y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]; colors = np.ones(phi_slice.shape + (3,)); abs_phi = np.abs(phi_slice)
    small = abs_phi <= threshold_1
    a_small = abs_phi[small] / threshold_1
    pos_small = small & (phi_slice >= 0); neg_small = small & (phi_slice < 0)
    colors[pos_small] = np.array([interpolate_color((1,1,1), mid_red, a) for a in a_small[phi_slice[small]>=0]]).reshape(-1,3)
    colors[neg_small] = np.array([interpolate_color((1,1,1), mid_blue, a) for a in a_small[phi_slice[small]<0]]).reshape(-1,3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = (abs_phi[med] - threshold_1) / (threshold_2 - threshold_1)
    pos_med = med & (phi_slice > 0); neg_med = med & (phi_slice < 0)
    colors[pos_med] = np.array([interpolate_color(mid_red, final_red, a) for a in a_med[phi_slice[med]>0]]).reshape(-1,3)
    colors[neg_med] = np.array([interpolate_color(mid_blue, final_blue, a) for a in a_med[phi_slice[med]<0]]).reshape(-1,3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red; colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1, 3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5); plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()


def save_slice_plot_new_sqrt(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6,6)); plt.title(f"t = {step*dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx); y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]; colors = np.ones(phi_slice.shape + (3,)); abs_phi = np.abs(phi_slice)
    small = abs_phi <= threshold_1
    a_small = np.sqrt(abs_phi[small] / threshold_1)
    pos_small = small & (phi_slice >= 0); neg_small = small & (phi_slice < 0)
    colors[pos_small] = np.array([interpolate_color((1,1,1), mid_red, a) for a in a_small[phi_slice[small]>=0]]).reshape(-1,3)
    colors[neg_small] = np.array([interpolate_color((1,1,1), mid_blue, a) for a in a_small[phi_slice[small]<0]]).reshape(-1,3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = np.sqrt((abs_phi[med] - threshold_1) / (threshold_2 - threshold_1))
    pos_med = med & (phi_slice > 0); neg_med = med & (phi_slice < 0)
    colors[pos_med] = np.array([interpolate_color(mid_red, final_red, a) for a in a_med[phi_slice[med]>0]]).reshape(-1,3)
    colors[neg_med] = np.array([interpolate_color(mid_blue, final_blue, a) for a in a_med[phi_slice[med]<0]]).reshape(-1,3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red; colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1,3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5); plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()


def save_slice_plot_new_log(phi, dt, Lx, Ly, Nx, Ny, step, z_slice, threshold_1, threshold_2, count, output_dir="plots", log_scale=10.0):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6,6)); plt.title(f"t = {step*dt:.2f}")
    x_coords = np.linspace(-Lx, Lx, Nx); y_coords = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    phi_slice = phi[:, :, z_slice]; colors = np.ones(phi_slice.shape + (3,))
    abs_phi = np.abs(phi_slice); tiny = threshold_1 / 100.0
    small = (abs_phi >= tiny) & (abs_phi <= threshold_1)
    val_small = abs_phi[small]; val_range = threshold_1 - tiny
    def log_map(v): return np.log(1 + log_scale*(v - tiny)) / np.log(1 + log_scale*val_range)
    a_small = np.array([log_map(v) for v in val_small])
    pos_small = small & (phi_slice >= 0); neg_small = small & (phi_slice < 0)
    colors[pos_small] = np.array([interpolate_color((1,1,1), mid_red, a) for a in a_small[phi_slice[small]>=0]]).reshape(-1,3)
    colors[neg_small] = np.array([interpolate_color((1,1,1), mid_blue, a) for a in a_small[phi_slice[small]<0]]).reshape(-1,3)
    med = (abs_phi > threshold_1) & (abs_phi <= threshold_2)
    a_med = (abs_phi[med] - threshold_1) / (threshold_2 - threshold_1)
    pos_med = med & (phi_slice > 0); neg_med = med & (phi_slice < 0)
    colors[pos_med] = np.array([interpolate_color(mid_red, final_red, a) for a in a_med[phi_slice[med]>0]]).reshape(-1,3)
    colors[neg_med] = np.array([interpolate_color(mid_blue, final_blue, a) for a in a_med[phi_slice[med]<0]]).reshape(-1,3)
    large = abs_phi > threshold_2
    colors[large & (phi_slice > 0)] = final_red; colors[large & (phi_slice < 0)] = final_blue
    Xf = X.ravel(); Yf = Y.ravel(); Cf = colors.reshape(-1,3)
    plt.scatter(Xf, Yf, c=Cf, marker='o', s=5); plt.xlim(-Lx, Lx); plt.ylim(-Ly, Ly); plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"slice_plot_{count}.png")); plt.close()
