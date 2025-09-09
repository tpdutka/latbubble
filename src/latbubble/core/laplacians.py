# laplacians.py
import numpy as np

# ======================== Periodic ===================================

# errors at O(a^4)
def laplacian_Oa4(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2)) 
    def lap_at_phi(phi):
        phi_extended = np.pad(phi, pad_width=2, mode='wrap')

        lap_phi = (
            Z_phix * (-np.roll(phi_extended, -2, axis=0) + 16 * np.roll(phi_extended, -1, axis=0)
                      - 30 * phi_extended + 16 * np.roll(phi_extended, 1, axis=0)
                      - np.roll(phi_extended, 2, axis=0)) / (12 * dx**2)
            +
            Z_phiy * (-np.roll(phi_extended, -2, axis=1) + 16 * np.roll(phi_extended, -1, axis=1)
                      - 30 * phi_extended + 16 * np.roll(phi_extended, 1, axis=1)
                      - np.roll(phi_extended, 2, axis=1)) / (12 * dy**2)
            +
            Z_phiz * (-np.roll(phi_extended, -2, axis=2) + 16 * np.roll(phi_extended, -1, axis=2)
                      - 30 * phi_extended + 16 * np.roll(phi_extended, 1, axis=2)
                      - np.roll(phi_extended, 2, axis=2)) / (12 * dz**2)
        )

        return lap_phi[2:-2, 2:-2, 2:-2]
    return lap_at_phi

# errors at O(a^2)
def laplacian_Oa2(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2)) 
    def lap_at_phi(phi):
        phi_extended = np.pad(phi, pad_width=1, mode='wrap')

        lap_phi = (
            Z_phix * (np.roll(phi_extended, -1, axis=0) - 2 * phi_extended + np.roll(phi_extended, 1, axis=0)) / dx**2
            +
            Z_phiy * (np.roll(phi_extended, -1, axis=1) - 2 * phi_extended + np.roll(phi_extended, 1, axis=1)) / dy**2
            +
            Z_phiz * (np.roll(phi_extended, -1, axis=2) - 2 * phi_extended + np.roll(phi_extended, 1, axis=2)) / dz**2
        )

        return lap_phi[1:-1, 1:-1, 1:-1]
    return lap_at_phi

# ===================== Neumann BC (∂φ/∂n = 0) =====================

def laplacian_Oa4_neumann(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2))
    def lap_at_phi(phi):
        # 2-layer padding for O(a^4) stencil, Neumann via 'edge'
        phi_extended = np.pad(phi, pad_width=2, mode='edge')

        lap_phi = (
            Z_phix * (-np.roll(phi_extended, -2, axis=0) + 16*np.roll(phi_extended, -1, axis=0)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=0)
                      - np.roll(phi_extended, 2, axis=0)) / (12 * dx**2)
            +
            Z_phiy * (-np.roll(phi_extended, -2, axis=1) + 16*np.roll(phi_extended, -1, axis=1)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=1)
                      - np.roll(phi_extended, 2, axis=1)) / (12 * dy**2)
            +
            Z_phiz * (-np.roll(phi_extended, -2, axis=2) + 16*np.roll(phi_extended, -1, axis=2)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=2)
                      - np.roll(phi_extended, 2, axis=2)) / (12 * dz**2)
        )
        return lap_phi[2:-2, 2:-2, 2:-2]
    return lap_at_phi


def laplacian_Oa2_neumann(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2))
    def lap_at_phi(phi):
        # 1-layer padding for O(a^2) stencil, Neumann via 'edge'
        phi_extended = np.pad(phi, pad_width=1, mode='edge')

        lap_phi = (
            Z_phix * (np.roll(phi_extended, -1, axis=0) - 2*phi_extended + np.roll(phi_extended, 1, axis=0)) / dx**2
            +
            Z_phiy * (np.roll(phi_extended, -1, axis=1) - 2*phi_extended + np.roll(phi_extended, 1, axis=1)) / dy**2
            +
            Z_phiz * (np.roll(phi_extended, -1, axis=2) - 2*phi_extended + np.roll(phi_extended, 1, axis=2)) / dz**2
        )
        return lap_phi[1:-1, 1:-1, 1:-1]
    return lap_at_phi


# ===================== Dirichlet BC (φ = 0) =====================

def laplacian_Oa4_dirichlet(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2))
    def lap_at_phi(phi):
        # 2-layer padding with zeros outside domain for Dirichlet
        phi_extended = np.pad(phi, pad_width=2, mode='constant', constant_values=0.0)

        lap_phi = (
            Z_phix * (-np.roll(phi_extended, -2, axis=0) + 16*np.roll(phi_extended, -1, axis=0)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=0)
                      - np.roll(phi_extended, 2, axis=0)) / (12 * dx**2)
            +
            Z_phiy * (-np.roll(phi_extended, -2, axis=1) + 16*np.roll(phi_extended, -1, axis=1)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=1)
                      - np.roll(phi_extended, 2, axis=1)) / (12 * dy**2)
            +
            Z_phiz * (-np.roll(phi_extended, -2, axis=2) + 16*np.roll(phi_extended, -1, axis=2)
                      - 30*phi_extended + 16*np.roll(phi_extended, 1, axis=2)
                      - np.roll(phi_extended, 2, axis=2)) / (12 * dz**2)
        )
        return lap_phi[2:-2, 2:-2, 2:-2]
    return lap_at_phi


def laplacian_Oa2_dirichlet(dx, dy, dz, C2, lam_lam, NNLO):
    Z_phix = 1 + NNLO * ((C2 * lam_lam**2 * dx**2) / (96 * np.pi**2))
    Z_phiy = 1 + NNLO * ((C2 * lam_lam**2 * dy**2) / (96 * np.pi**2))
    Z_phiz = 1 + NNLO * ((C2 * lam_lam**2 * dz**2) / (96 * np.pi**2))
    def lap_at_phi(phi):
        # 1-layer padding with zeros outside domain for Dirichlet
        phi_extended = np.pad(phi, pad_width=1, mode='constant', constant_values=0.0)

        lap_phi = (
            Z_phix * (np.roll(phi_extended, -1, axis=0) - 2*phi_extended + np.roll(phi_extended, 1, axis=0)) / dx**2
            +
            Z_phiy * (np.roll(phi_extended, -1, axis=1) - 2*phi_extended + np.roll(phi_extended, 1, axis=1)) / dy**2
            +
            Z_phiz * (np.roll(phi_extended, -1, axis=2) - 2*phi_extended + np.roll(phi_extended, 1, axis=2)) / dz**2
        )
        return lap_phi[1:-1, 1:-1, 1:-1]
    return lap_at_phi