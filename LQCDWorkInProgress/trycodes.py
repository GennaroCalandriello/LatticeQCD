import numpy as np
from numba import njit


from module.functionsu3 import *

# Define the gamma matrices
gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma_1 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]])

gamma = np.zeros(((4, 4, 4)), dtype=complex)

gamma[0] = gamma_0
gamma[1] = gamma_1
gamma[2] = gamma_2
gamma[3] = gamma_3

N = 5
Nx, Ny, Nz, Nt = N, N, N, N  # possibility to asymmetric time extensions
Dirac = 4
color = 3


@njit()
def quarkfield(quark, forloops):

    """Each flavor of fermion (quark) has 4 space-time indices, 3 color
     indices and 4 spinor (Dirac) indices"""

    if forloops:
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for t in range(Nt):
                        for spinor in range(Dirac):
                            for Nc in range(color):
                                quark[x, y, z, t, spinor, Nc] = complex(
                                    np.random.rand(), np.random.rand()
                                )
    # oppure più banalmente
    if not forloops:
        quark = np.random.rand(Nx, Ny, Nz, Nt, Dirac, color) + 1j * np.random.rand(
            Nx, Ny, Nz, Nt, color, Dirac
        )

    return quark


@njit()
def initializefield(U):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = SU3SingleMatrix() * complex(
                            np.random.rand(), np.random.rand()
                        )

    return U


def DiracMatrix(U, psi, D):

    m = 0.8

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    D[x, y, z, t] += m
                    for alpha in range(Dirac):
                        for beta in range(Dirac):
                            for a in range(color):
                                for b in range(color):
                                    for mu in range(4):
                                        a_mu = [0, 0, 0, 0]
                                        a_mu[mu] = 1
                                        D[x, y, z, t, alpha, beta, a, b] += 0.5 * (
                                            gamma[mu][alpha, beta]
                                            * U[x, y, z, t, mu, a, b]
                                            * psi[
                                                (x + a_mu[0]) % Nx,
                                                (y + a_mu[1]) % Ny,
                                                (z + a_mu[2]) % Nz,
                                                (t + a_mu[3]) % Nt,
                                                alpha,
                                                a,
                                            ]
                                            - U[
                                                (x - a_mu[0]) % Nx,
                                                (y - a_mu[1]) % Ny,
                                                (z - a_mu[2]) % Nz,
                                                (t - a_mu[3]) % Nt,
                                                mu,
                                                a,
                                                b,
                                            ]
                                            .conj()
                                            .T
                                            * psi[
                                                (x - a_mu[0]) % Nx,
                                                (y - a_mu[1]) % Ny,
                                                (z - a_mu[2]) % Nz,
                                                (t - a_mu[3]) % Nt,
                                                beta,
                                                b,
                                            ]
                                        )

    return D


def DiracMatrix128(U, psi, D):

    m = 2

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for alpha in range(Dirac):
                        for beta in range(Dirac):
                            for a in range(color):
                                for b in range(color):
                                    for mu in range(4):
                                        a_mu = [0, 0, 0, 0]
                                        a_mu[mu] = 1
                                        D[x, y, z, t, alpha, beta, a, b] += 0.5 * (
                                            gamma[mu][alpha, beta]
                                            * U[x, y, z, t, mu, a, b]
                                            * psi[
                                                (x + a_mu[0]) % Nx,
                                                (y + a_mu[1]) % Ny,
                                                (z + a_mu[2]) % Nz,
                                                (t + a_mu[3]) % Nt,
                                                alpha,
                                                a,
                                            ]
                                            - U[
                                                (x - a_mu[0]) % Nx,
                                                (y - a_mu[1]) % Ny,
                                                (z - a_mu[2]) % Nz,
                                                (t - a_mu[3]) % Nt,
                                                mu,
                                                a,
                                                b,
                                            ]
                                            .conj()
                                            .T
                                            * psi[
                                                (x - a_mu[0]) % Nx,
                                                (y - a_mu[1]) % Ny,
                                                (z - a_mu[2]) % Nz,
                                                (t - a_mu[3]) % Nt,
                                                beta,
                                                b,
                                            ]
                                            + m
                                        )

    return D


def LinearlyIndependent(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j:
                inner_product = np.inner(matrix[:, i], matrix[:, j])
                norm_i = np.linalg.norm(matrix[:, i])
                norm_j = np.linalg.norm(matrix[:, j])

                print("I: ", matrix[:, i])
                print("J: ", matrix[:, j])
                print("Prod: ", inner_product)
                print("Norm i: ", norm_i)
                print("Norm j: ", norm_j)
                if np.abs(inner_product - norm_j * norm_i) < 1e-5:
                    print("Dependent")
                else:
                    print("Independent")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    U = np.zeros((Nx, Ny, Nz, Nt, 4, su3, su3), dtype=complex)
    psi = np.zeros((Nx, Ny, Nz, Nt, Dirac, color), complex)
    D = np.zeros((Nx, Ny, Nz, Nt, Dirac, Dirac, color, color), complex)
    U = initializefield(U)
    psi = quarkfield(psi, True)

    D = DiracMatrix(U, psi, D)
    # print(D)

    eig, _ = np.linalg.eig(D)
    eig = eig.flatten()
    spacing = []
    # dls = D[0, 0, 0, 0, 0, 0]
    # print(dls)
    # LinearlyIndependent(dls)
    for i in range(1, len(eig) - 1):
        spacing.append(min(eig[i + 1] - eig[i], eig[i] - eig[i - 1]))
    print(np.mean(spacing))
    plt.figure()
    plt.hist(spacing, 120)
    plt.show()

