import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from scipy.linalg import expm
from numba import njit
from torch import rand

from module.anderson2d import *

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

"""----------------------------------------------------------------------------------------------------------------
references:
1. https://arxiv.org/pdf/cond-mat/0410190.pdf"""


def SU2SingleMatrix(epsilon=0.2):
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])

    r0 = np.random.uniform(-0.5, 0.5)
    x0 = np.sign(r0) * np.sqrt(1 - epsilon**2)

    r = np.random.random((3)) - 0.5
    x = epsilon * r / np.linalg.norm(r)

    SU2Matrix = x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz

    return SU2Matrix


def Wilson():
    U = np.zeros((L, L, L, L, 4))
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        U[x, y, x, y, mu] = np.random.uniform(-1, 1)
    return U


@njit()
def PDF(theta):
    """Probability density function."""
    return np.sin(2 * theta)


def geometry(L):
    npp = [i + 1 for i in range(0, L)]
    nmm = [i - 1 for i in range(0, L)]
    npp[L - 1] = 0
    nmm[0] = L - 1

    return np.array(npp), np.array(nmm)


@njit()
def metropolisHastings(n_samples=1200000, theta_init=0, step_size=0.05):
    """
    ----------------------------------------------------------------
    Metropolis sampling algorithm.
    ----------------------------------------------------------------
    njit() test: passed
    Correctly samples the PDF increasing step_size (set it small as 0.01)."""

    samples = np.zeros(n_samples, dtype=np.float64)
    theta_old = theta_init
    p_old = PDF(theta_old)
    n_hits = 10
    for i in range(n_samples):
        for _ in range(n_hits):
            theta_new = np.random.uniform(theta_old - step_size, theta_old + step_size)
            if theta_new < 0 or theta_new > np.pi / 2:
                # If the proposal is outside the interval [0, pi/2], reject it
                samples[i] = theta_old
            else:
                p_new = PDF(theta_new)
                if p_new > p_old or np.random.rand() < p_new / p_old:
                    # If the proposal is more likely or we're moving to a less likely state, accept it
                    samples[i] = theta_new
                    theta_old = theta_new
                    p_old = p_new
                else:
                    # Otherwise, reject it
                    samples[i] = theta_old

    return samples


def cdf(beta):
    return -0.5 * np.cos(2 * beta) + 0.5


def find_inverse_cdf(y_target):
    # Root finding algorithm to get inverse CDF
    return optimize.root(lambda beta: cdf(beta) - y_target, 0).x[0]


def sample_beta():
    y_random = np.random.uniform(
        0, np.pi / 2
    )  # Random number from uniform distribution
    beta_sample = find_inverse_cdf(y_random)
    return beta_sample


@njit
def shuffle_array(x):
    for i in range(len(x) - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        x[i], x[j] = x[j], x[i]
    return x


@njit()
def Rpar():
    """-----------------Rpar-----------------
    Generates the matrices to construct R(ij)
    --------------------------------------
    njit() test: passed"""

    betaij = np.zeros((L, L), dtype=np.complex128)
    alphaij = betaij.copy()
    gammaij = betaij.copy()

    metropolis = True
    if metropolis:
        metro = metropolisHastings()
        metro = metro[40000:]  # delete thermalization
        metro = shuffle_array(metro)

        for i in range(L):
            for j in range(L):
                betaij[i, j] = metro[i * L + j]
    # else:
    #     for i in range(L):
    #         for j in range(L):
    #             betaij[i, j] = sample_beta()
    for i in range(L):
        for j in range(L):
            alphaij[i, j] = np.random.uniform(0, 2 * np.pi)
            gammaij[i, j] = np.random.uniform(0, 2 * np.pi)

    return betaij, alphaij, gammaij


@njit()
def R(betaij, alphaij, gammaij, dir):
    """-----------------R-------------------------------------------
    Generates the R(ij) matrix
    ----------------------------------------------------------------
    njit() test: passed----------------------------------------------
    Rij is symplectic, verified with is_symplectic()-----------------"""

    if dir == "y":
        c01 = -1j
        c10 = 1j
        c00 = 1
        c11 = -1
    if dir == "x":
        c01 = 1
        c10 = 1
        c00 = 1
        c11 = 1

    Rij = np.zeros((2, 2), dtype=np.complex128)
    Rij[0, 0] = c00 * np.exp(1j * alphaij) * np.cos(betaij)
    Rij[0, 1] = c01 * np.exp(1j * gammaij) * np.sin(betaij)
    Rij[1, 0] = c10 * (-np.exp(-1j * gammaij) * np.sin(betaij))
    Rij[1, 1] = c11 * np.exp(-1j * alphaij) * np.cos(betaij)

    return Rij


@njit()
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


@njit
def expMatrix(A):
    """
    Approximate the matrix exponential using a Taylor series.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to exponentiate.
    n : int
        The order of the Taylor series approximation.

    Returns
    -------
    numpy.ndarray
        The approximated matrix exponential of A.
    """
    n = 10
    shape = len(A[0])
    expA = np.zeros_like(A, dtype=np.complex128)
    term = np.eye(shape, dtype=np.complex128)

    for i in range(n):
        expA += term
        term = (term @ A) / factorial(i + 1)

    return expA


@njit()
def QuaternionMatrix(z0, z1, z2, z3):
    """----------------------------------------------------------------------------------------------------------------
    reference: {https://journals.aps.org/prb/pdf/10.1103/PhysRevB.40.5325}
    ----------------------------------------------------------------------------------------------------------------
    """
    z = np.zeros((2, 2), dtype=np.complex128)
    Id = np.eye(2, dtype=np.complex128)
    tau_1 = np.array([[1j, 0], [0, -1j]], dtype=np.complex128)  # Spin operator
    tau_2 = np.array([[0, -1], [1, 0]], dtype=np.complex128)  # Spin operator
    tau_3 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)  # Spin operator
    z = z0 * Id + z1 * tau_1 + z2 * tau_2 + z3 * tau_3

    return z


@njit()
def Hamiltonian():
    """-----------------Hamiltonian-----------------
    Generates the Hamiltonian of the system
    --------------------------------------
    njit() test: passed"""

    theta = np.pi / 6
    H = np.zeros((2 * L * L, 2 * L * L), dtype=np.complex128)  # Initialize Hamiltonian

    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # Spin operator
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Spin operator
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)  # Spin operator
    Pauli = [sigma_x, sigma_y, sigma_z]
    betaij, alphaij, gammaij = Rpar()
    # On-site disorder term

    for i in range(L):
        for j in range(L):
            idx = 2 * (i * L + j)  # Calculate the index in the Hamiltonian
            disorder = np.random.uniform(-W / 2, W / 2)  # Random disorder term

            rand = np.random.uniform(-W / 2, W / 2, 4)
            # quat = QuaternionMatrix(rand[0], rand[1], rand[2], rand[3])
            H[idx : idx + 2, idx : idx + 2] = sigma_z * disorder
            # is_symplectic(quat)  # on diagonal correct

    for i in range(L):
        for j in range(L):
            idx1 = i * L + j
            # Nearest neighbors (with periodic boundary conditions)
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            neighbors = [(n_i % L, n_j % L) for n_i, n_j in neighbors]

            for nx, ny in neighbors:
                if nx == i - 1 and ny == j:
                    dir = "x"
                if nx == i + 1 and ny == j:
                    dir = "x"
                if nx == i and ny == j - 1:
                    dir = "y"
                if nx == i and ny == j + 1:
                    dir = "y"
                idx2 = nx * L + ny

                H[idx1 * 2 : (idx1 + 1) * 2, idx2 * 2 : (idx2 + 1) * 2] += R(
                    betaij[i, j], alphaij[i, j], gammaij[i, j], dir
                )
                #     @ R(betaij[i, j], alphaij[i, j], gammaij[i, j], dir).conj().T
                # )
    print(H)
    return H


def main():
    H = Hamiltonian()

    print("---- Hamiltonian generated! ----", H.shape)

    (energy_levels, eigenstates) = linalg.eigh(H)

    # The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
    # Thus, sort them
    idx = np.argsort(energy_levels)
    # energy_levels = energy_levels[idx]

    eigenstates = eigenstates[:, idx]
    unfolded = np.array(unfold_spectrum(energy_levels)[1])
    p = distribution(unfolded, "GUE")

    plt.figure()
    plt.hist(
        unfolded,
        bins=FreedmanDiaconis(unfolded),
        density=True,
        histtype="step",
        color="b",
        label="unfolded spectrum",
    )
    plt.plot(
        np.linspace(min(unfolded), max(unfolded), len(p)),
        distribution(unfolded, "GSE"),
        "--",
        label="GSE",
        color="b",
    )
    plt.plot(
        np.linspace(min(unfolded), max(unfolded), len(p)),
        distribution(unfolded, "GUE"),
        "--",
        label="GUE",
        color="r",
    )
    plt.legend()
    plt.show()


def is_symplectic(matrix):
    """----------------------------------------------------------------
    Check if a matrix is symplectic.
    -------------------------------------------------------------------
    """
    n = matrix.shape[0] // 2
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    sym = np.allclose(matrix.T @ J @ matrix, J)
    if sym:
        print("Symplectic")
    else:
        print("Not symplectic")


if __name__ == "__main__":
    main()
    # r = np.random.uniform(-W / 2, W / 2, 4)
    # q = QuaternionMatrix(r[0], r[1], r[2], r[3]) #quaterion matrx is not symplectic
    # is_symplectic(q)

    # b, a, g = Rpar()
    # Rij = R(b[0, 3], a[0, 3], g[0, 3], "x")
    # T = Rij.T
    # if T.any() == -1 * Rij.any():
    #     print("yesymp")
    # (is_symplectic(Rij))

    # H = Hamiltonian()
    # np.savetxt("H.txt", H)
    # samples = metropolisHastings()
    # samples = samples[100000:]
    # plt.hist(
    #     samples,
    #     bins=FreedmanDiaconis(samples),
    #     density=True,
    #     histtype="step",
    #     color="b",
    # )
    # plt.show()
    # t = PDF(np.linspace(0, np.pi / 2, 1000))
    # plt.plot(np.linspace(0, np.pi / 2, 1000), t)
    # plt.show()

    """--------------------------------------------Appunti------------------------------------------------
    #General form of a quaternion

    # q = [ a + bi  c + di ]
    # [-c + di  a - bi ]
    -----------------------------------------------------------------------------------------------------"""
