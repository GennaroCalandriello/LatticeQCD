from __future__ import print_function
from tkinter import font
import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats, sparse
from scipy.sparse import linalg as lnlg
from numba import njit
from torch import exp_

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
"""____________________________________________________________________
-----------------------------------------------------------------------------------------
This script models localization in the 2d Anderson model with box disorder,
i.e. uncorrelated on-site energies w_n uniformly distributed in [-W/2,W/2].
The script diagonalizes the Hamiltonian for a system of finite size L, and periodic boundary conditions
Without disorder, the dispersion relation is energy=2*[cos(kx)+cos(ky)], with support on [-4,4],
so that, for a  finite size system, the eigenstates are plane waves with
kx=i*2*\pi/L, ky=j*2*\pi/L with i,j integers -L/2<i,j<=L/2.
Eigenstates with +/i,+/-j are degenerate, allowing to build symmetric and antisymmetric combinations which
are thus real wavefunctions.
In the presence of disorder, the eigenstates are localized.
The localization length is not known analytically, but huge for small W.
The script computes and prints the full energy spectum for a single realization of the disorder
It also prints the wavefunction of the state which has energy closest to the input parameter "energy"
# -----------------------------------------------------------------------------------------"""

L = 20  # System size
W = 18  # Disorder strength
energy = 0


# System parameters
N = 1000  # Number of sites in the system
mean = 0  # Mean of the normal distribution
std_dev = 1  # Standard deviation of the normal distribution


def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, potential, neigs, E0):
    """Solve in the following steps:
    1. Construct meshgrid
    2. Evaluate potential
    3. Construct the two part of the Hamiltonian, Hx, Hy, and the two identity matrices Ix, Iy for the Kronecker sum
    4. Find the eigenvalues and eigenfunctions
    Basically one can plot all |psi|^2 eigenvectors obtaining the chaotic structure for various potentials
    """

    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = potential(x, y)

    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / dx**2

    Hy = sparse.lil_matrix(np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i + 1, i] = -1
        Hy[i, i + 1] = -1
    Hy = Hy / dy**2

    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))

    # Kronecker sum
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)

    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]

    # convert to sparse csc matrix form and calculate eigenvalues
    H = H.tocsc()
    [eigenvalues, eigenstates] = lnlg.eigs(H, k=neigs, sigma=E0)

    return eigenvalues, eigenstates


def H_rho():
    # Draw random values from the normal distribution
    rho_values = W * np.random.normal(mean, std_dev, N)

    # Construct a diagonal matrix with these values
    rho_matrix = np.diag(rho_values)

    return rho_matrix


def cnDiff(eigs, kind):
    spacing = []

    if kind == "CN":
        for i in range(1, len(eigs) - 1):
            s = min(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1])
            spacing.append(s)
        return np.array(spacing)

    if kind == "FN":
        for i in range(1, len(eigs) - 1):
            s = max(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1])
            spacing.append(s)
        return np.array(spacing)

    if kind == "diff":
        return np.diff(eigs)


def PauliMatrices(i):
    if i == 0:
        sigma_i = np.array([[0, 1], [1, 0]])
    if i == 1:
        sigma_i = np.array([[0, -1j], [1j, 0]])
    if i == 2:
        sigma_i = np.array([[1, 0], [0, -1]])

    return sigma_i


def FreedmanDiaconis(spacings):
    # Compute IQR
    IQR = np.percentile(spacings, 75) - np.percentile(spacings, 25)

    # Compute bin width
    bin_width = 2 * IQR * (len(spacings) ** (-1 / 3))

    # Compute number of bins
    num_bins = np.ceil((max(spacings) - min(spacings)) / bin_width)

    return int(num_bins)


def distribution(sp, kind):
    """Plot theoretical distributions of GSE, GOE, GUE ensemble distributions picking the min and max values of the spacing array
    calculated in the main program"""
    s = np.linspace(0, max(sp), len(sp))
    p = np.zeros(len(s))

    if kind == "GOE":
        for i in range(len(p)):
            p[i] = np.pi / 2 * s[i] * np.exp(-np.pi / 4 * s[i] ** 2)

    if kind == "GUE":
        for i in range(len(p)):
            p[i] = (32 / np.pi**2) * s[i] ** 2 * np.exp(-4 / np.pi * s[i] ** 2)

    if kind == "GSE":
        for i in range(len(p)):
            p[i] = (
                2**18
                / (3**6 * np.pi**3)
                * s[i] ** 4
                * np.exp(-(64 / (9 * np.pi)) * s[i] ** 2)
            )
    if kind == "Poisson":
        for i in range(len(p)):
            p[i] = np.exp(-s[i])

    if kind == "GOE FN":  # lasciamo perdere va...
        a = (27 / 8) * np.pi
        for i in range(len(p)):
            p[i] = (
                (a / np.pi)
                * s[i]
                * np.exp(-2 * a * s[i] ** 2)
                * (
                    np.pi
                    * np.exp((3 * a / 2) * s[i] ** 2)
                    * (a * s[i] ** 2 - 3)
                    * (
                        math.erf(np.sqrt(a / 6) * s[i])
                        - math.erf(np.sqrt(3 * a / 2) * s[i])
                        + np.sqrt(6 * np.pi * a)
                        * s[i]
                        * (np.exp((4 * a / 3) * s[i] ** 2) - 3)
                    )
                )
            )

    return p


def unfold_spectrum(eigvals, kind):
    # Perform kernel density estimation
    eigvals = np.sort(eigvals)
    kde = stats.gaussian_kde(eigvals)

    # Create a regular grid on which to evaluate the KDE
    eigvals_grid = np.linspace(min(eigvals), max(eigvals), len(eigvals))

    # Evaluate the KDE on the grid
    density = kde(eigvals_grid)

    # Normalize the density to the number of eigenvalues
    density /= sum(density)

    # Calculate the cumulative distribution
    cum_density = np.cumsum(density)

    # The unfolding function is the inverse of the cumulative distribution
    unfold_func = lambda x: np.interp(x, eigvals_grid, cum_density)

    # Apply the unfolding function to the eigenvalues
    unfolded_eigvals = unfold_func(eigvals)
    unfolded_spacings = cnDiff(unfolded_eigvals, kind)  ##np.diff(unfolded_eigvals)

    return unfolded_eigvals, unfolded_spacings / np.mean(unfolded_spacings)


# Generate a disordered sequence of on-site energies in the array "disorder"


def generate_disorder(L, W):
    disorder = W * ((np.random.uniform(size=L * L)).reshape((L, L)) - 0.5)
    return disorder


import numpy as np
import matplotlib.pyplot as plt


def oneD_to_threeD(psi):
    """Transform vectors in matrices"""

    count = 0
    PSI = np.zeros([L, L, L], dtype="complex")
    for i in range(L):
        for j in range(L):
            for k in range(L):
                PSI[i, j, k] = psi[count]
                count += 1
    return PSI


def plottingEigenstates(eigenstates):
    for n in range(4):
        psi = eigenstates[:, n]
        PSI = oneD_to_threeD(psi)
        PSI = np.abs(PSI) ** 2
        plt.pcolormesh(np.flipud(PSI), cmap="terrain")
        plt.axis("equal")
        plt.axis("off")
        plt.show()


def localization_length(psi, lattice):
    """Calculate the localization length of a wave function on a lattice."""

    # Calculate the center of mass of the absolute square of the wave function
    r_cm = np.sum(np.abs(psi) ** 2 * lattice, axis=0) / np.sum(np.abs(psi) ** 2)

    # Calculate the second moment of the absolute square of the wave function
    r2_moment = np.sum(
        np.abs(psi) ** 2 * np.sum((lattice - r_cm) ** 2, axis=-1)
    ) / np.sum(np.abs(psi) ** 2)

    # The localization length is approximately the square root of the second moment
    xi = np.sqrt(r2_moment)

    return xi


def plotting(unf, kind, w):
    p = distribution(unf, kind)
    poiss = distribution(unf, "Poisson")
    plt.figure()
    plt.hist(
        unf,
        bins=FreedmanDiaconis(unf),
        histtype="step",
        color="blue",
        density=True,
        label="Numerical",
    )
    plt.plot(np.linspace(min(unf), max(unf), len(p)), p, "r--", label=kind)
    plt.plot(np.linspace(min(unf), max(unf), len(poiss)), poiss, "g--", label="Poisson")
    plt.xlabel("s", fontsize=14)
    plt.ylabel("P(s)", fontsize=14)
    plt.title(f"Distribution of spacings for {kind} ensemble, W = {w}", fontsize=16)
    plt.legend()
    plt.show()
