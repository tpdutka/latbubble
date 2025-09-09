# src/latticesim/utils/noise.py
import numpy as np

def compute_noise(D, dx, dy, dz, dt, shape):
    return np.random.normal(0, np.sqrt(D / (dt * dx * dy * dz)), size=shape)

def momentum_refresh(pi, gamma, dx, dt, noise_term):
    eta = 1 - np.exp(-2 * gamma * dt)
    return np.sqrt(eta) * (noise_term / dx**(3/2)) + np.sqrt(1 - eta) * pi