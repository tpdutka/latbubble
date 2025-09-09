# src/latticesim/core/integrators.py
import numpy as np
from latticesim.utils.noise import compute_noise, momentum_refresh

def _rhs(phi, phi_t, gamma, potential, laplacian, noise_term, use_refresh):
    if use_refresh:
        # OU handled separately -> Hamiltonian part only
        return laplacian(phi) - potential(phi)
    else:
        return laplacian(phi) - gamma * phi_t - potential(phi) + noise_term

def euler(phi, phi_t, dt, dx, dy, dz, D, gamma, potential, laplacian, mom_refresh):
    noise = 0.0
    if not mom_refresh:
        noise = compute_noise(D, dx, dy, dz, dt, phi.shape)

    k_phi   = phi_t
    k_phi_t = _rhs(phi, phi_t, gamma, potential, laplacian, noise, mom_refresh)

    phi_next   = phi   + dt * k_phi
    phi_t_next = phi_t + dt * k_phi_t

    if mom_refresh:
        z = np.random.standard_normal(phi.shape)
        phi_t_next = momentum_refresh(phi_t_next, gamma, dx, dt, z)
    return phi_next, phi_t_next

def rk2(phi, phi_t, dt, dx, dy, dz, D, gamma, potential, laplacian, mom_refresh):
    noise = 0.0
    if not mom_refresh:
        noise = compute_noise(D, dx, dy, dz, dt, phi.shape)

    k1_phi   = phi_t
    k1_phi_t = _rhs(phi, phi_t, gamma, potential, laplacian, noise, mom_refresh)

    phi_mid   = phi   + 0.5 * dt * k1_phi
    phi_t_mid = phi_t + 0.5 * dt * k1_phi_t

    k2_phi   = phi_t_mid
    k2_phi_t = _rhs(phi_mid, phi_t_mid, gamma, potential, laplacian, noise, mom_refresh)

    phi_next   = phi   + dt * k2_phi
    phi_t_next = phi_t + dt * k2_phi_t

    if mom_refresh:
        z = np.random.standard_normal(phi.shape)
        phi_t_next = momentum_refresh(phi_t_next, gamma, dx, dt, z)
    return phi_next, phi_t_next

def rk4(phi, phi_t, dt, dx, dy, dz, D, gamma, potential, laplacian, mom_refresh):
    noise = 0.0
    if not mom_refresh:
        noise = compute_noise(D, dx, dy, dz, dt, phi.shape)

    k1_phi   = phi_t
    k1_phi_t = _rhs(phi, phi_t, gamma, potential, laplacian, noise, mom_refresh)

    phi_2   = phi   + 0.5 * dt * k1_phi
    phi_t_2 = phi_t + 0.5 * dt * k1_phi_t
    k2_phi   = phi_t_2
    k2_phi_t = _rhs(phi_2, phi_t_2, gamma, potential, laplacian, noise, mom_refresh)

    phi_3   = phi   + 0.5 * dt * k2_phi
    phi_t_3 = phi_t + 0.5 * dt * k2_phi_t
    k3_phi   = phi_t_3
    k3_phi_t = _rhs(phi_3, phi_t_3, gamma, potential, laplacian, noise, mom_refresh)

    phi_4   = phi   + dt * k3_phi
    phi_t_4 = phi_t + dt * k3_phi_t
    k4_phi   = phi_t_4
    k4_phi_t = _rhs(phi_4, phi_t_4, gamma, potential, laplacian, noise, mom_refresh)

    phi_next   = phi   + (dt / 6.0) * (k1_phi   + 2 * k2_phi   + 2 * k3_phi   + k4_phi)
    phi_t_next = phi_t + (dt / 6.0) * (k1_phi_t + 2 * k2_phi_t + 2 * k3_phi_t + k4_phi_t)

    if mom_refresh:
        z = np.random.standard_normal(phi.shape)
        phi_t_next = momentum_refresh(phi_t_next, gamma, dx, dt, z)
    return phi_next, phi_t_next

def leapfrog(phi, phi_t, dt, dx, dy, dz, D, gamma, potential, laplacian, mom_refresh):
    noise = 0.0
    if not mom_refresh:
        noise = compute_noise(D, dx, dy, dz, dt, phi.shape)

    # half-kick
    phi_t_half = phi_t + 0.5 * dt * _rhs(phi, phi_t, gamma, potential, laplacian, noise, mom_refresh)
    # drift
    phi_next = phi + dt * phi_t_half
    # half-kick
    phi_t_next = phi_t_half + 0.5 * dt * _rhs(phi_next, phi_t_half, gamma, potential, laplacian, noise, mom_refresh)

    if mom_refresh:
        z = np.random.standard_normal(phi.shape)
        phi_t_next = momentum_refresh(phi_t_next, gamma, dx, dt, z)
    return phi_next, phi_t_next

def forest_ruth(phi, phi_t, dt, dx, dy, dz, D, gamma, potential, laplacian, mom_refresh):
    # Symplectic coefficients
    lam1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    lam2 = 1.0 - 2.0 * lam1

    def a(phi, phi_t):  # drift
        return phi_t

    def b(phi, phi_t):  # kick RHS
        if mom_refresh:
            return laplacian(phi) - potential(phi)
        else:
            noise = compute_noise(D, dx, dy, dz, dt, phi.shape)
            return laplacian(phi) - gamma * phi_t - potential(phi) + noise

    # KDK, KDK, KDK with (lam1, lam2, lam1)
    phi_t = phi_t + 0.5 * lam1 * dt * b(phi, phi_t)
    phi   = phi   +       lam1 * dt * a(phi, phi_t)
    phi_t = phi_t + 0.5 * lam1 * dt * b(phi, phi_t)

    phi_t = phi_t + 0.5 * lam2 * dt * b(phi, phi_t)
    phi   = phi   +       lam2 * dt * a(phi, phi_t)
    phi_t = phi_t + 0.5 * lam2 * dt * b(phi, phi_t)

    phi_t = phi_t + 0.5 * lam1 * dt * b(phi, phi_t)
    phi   = phi   +       lam1 * dt * a(phi, phi_t)
    phi_t = phi_t + 0.5 * lam1 * dt * b(phi, phi_t)

    if mom_refresh:
        z = np.random.standard_normal(phi.shape)
        phi_t = momentum_refresh(phi_t, gamma, dx, dt, z)
    return phi, phi_t