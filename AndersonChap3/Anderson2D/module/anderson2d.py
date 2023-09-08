from __future__ import print_function
import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats, sparse
from scipy.sparse import linalg as lnlg


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

L = 34
W = 7.9
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


def cnDiff(eigs):
    spacing = []
    kind = "diff"

    if kind == "CN":
        for i in range(1, len(eigs) - 1):
            s = min(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1])
            spacing.append(s)
        return np.array(spacing)

    if kind == "FN":
        for i in range(1, len(eigs) - 1):
            s = min(eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1])
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


def unfold_spectrum(eigvals):
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
    unfolded_spacings = cnDiff(unfolded_eigvals)  ##np.diff(unfolded_eigvals)

    return unfolded_eigvals, unfolded_spacings / np.mean(unfolded_spacings)


# Generate a disordered sequence of on-site energies in the array "disorder"
def generate_disorder(L, W):
    disorder = W * ((np.random.uniform(size=L * L)).reshape((L, L)) - 0.5)
    return disorder


def GenerateDisorderSpinOrbit(L, W):
    disorder = W * ((np.random.uniform(size=L * L)).reshape((L, L)) - 0.5)

    SpinCoupledDisorder = 0 + 0j  # np.zeros((2 * L, 2 * L), dtype=complex)
    for i in range(3):
        SpinCoupledDisorder += np.kron(disorder, PauliMatrices(i))

    return SpinCoupledDisorder


def generate_hamiltonian(L, W, magneticfield, symplectic=False):
    """Generate the Hamiltonian matrix for one realization of the random disorder. The value
    of magneticfield is the flux of B added as a phase factor to the hopping terms.
    The presence of B switches the ULSD from GOE to GUE."""

    if not symplectic:
        H = np.zeros((L * L, L * L), dtype=complex)
        disorder = generate_disorder(L, W)
        for i in range(L):
            ip1 = (i + 1) % L
            for j in range(L):
                expo = np.exp(
                    -2 * 1j * np.pi * magneticfield
                )  # add a phase factor depending on the magnetic flux value
                H[i * L + j, i * L + j] = (
                    disorder[i, j] * expo
                )  # diagonal elements H[i * L + j, i * L + j]

                jp1 = (j + 1) % L

                H[ip1 * L + j, i * L + j] = 1.0 * expo
                H[i * L + j, ip1 * L + j] = 1.0 * expo
                H[i * L + jp1, i * L + j] = 1.0 * expo
                H[i * L + j, i * L + jp1] = 1.0 * expo
        return H

    else:
        H = generate_symplectic_Hamiltonian(L, W, magneticfield=0.0)

    return H


def generate_symplectic_Hamiltonian(L, W, magneticfield=0.0):
    """Experimental use only!!!"""

    mu = 0.5
    # Define the Pauli matrices
    PauliMatricesArray = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    ]

    H = np.zeros((2 * L * L, 2 * L * L), dtype=complex)
    disorder = GenerateDisorderSpinOrbit(L, W)
    print("H, disorder", H.shape, disorder.shape)

    for i in range(L):
        ip1 = (i + 1) % L
        for j in range(L):
            # expo = np.exp(
            #     -2 * 1j * np.pi * magneticfield
            # )  # add a phase factor depending on the magnetic flux value
            expo = 1
            jp1 = (j + 1) % L

            for sigma in range(2):
                for sigma_prime in range(2):
                    # disorder term

                    H[2 * (i * L + j) + sigma, 2 * (i * L + j) + sigma_prime] = (
                        disorder[2 * i + sigma, 2 * j + sigma_prime] * expo
                    )
                    # hopping term

                    H[2 * (ip1 * L + j) + sigma, 2 * (i * L + j) + sigma_prime] = (
                        1 * expo * disorder[2 * ip1 + sigma, 2 * j + sigma_prime]
                    )
                    H[2 * (i * L + j) + sigma, 2 * (ip1 * L + j) + sigma_prime] = (
                        1 * expo * disorder[2 * ip1 + sigma, 2 * j + sigma_prime]
                    )
                    H[2 * (i * L + jp1) + sigma, 2 * (i * L + j) + sigma_prime] = (
                        1 * expo * disorder[2 * i + sigma, 2 * jp1 + sigma_prime]
                    )
                    H[2 * (i * L + j) + sigma, 2 * (i * L + jp1) + sigma_prime] = (
                        1 * expo * disorder[2 * i + sigma, 2 * jp1 + sigma_prime]
                    )

                    kvec = np.random.uniform(0, 1, 3) - 1 / 2
                    for k in range(3):
                        H[2 * (ip1 * L + j) + sigma, 2 * (i * L + j) + sigma_prime] += (
                            1j
                            * mu
                            * kvec[k]
                            * (PauliMatricesArray[k][sigma, sigma_prime])
                        )
                        H[2 * (i * L + j) + sigma, 2 * (ip1 * L + j) + sigma_prime] += (
                            1j
                            * mu
                            * kvec[k]
                            * (PauliMatricesArray[k][sigma, sigma_prime])
                        )
                        H[2 * (i * L + jp1) + sigma, 2 * (i * L + j) + sigma_prime] += (
                            1j
                            * mu
                            * kvec[k]
                            * (PauliMatricesArray[k][sigma, sigma_prime])
                        )
                        H[2 * (i * L + j) + sigma, 2 * (i * L + jp1) + sigma_prime] += (
                            1j
                            * mu
                            * kvec[k]
                            * (PauliMatricesArray[k][sigma, sigma_prime])
                        )

    return H


def main():
    variousB = False
    symplectic = True

    if not variousB:
        H = generate_hamiltonian(L, W, magneticfield=0.0, symplectic=symplectic)
        print(H.shape)
        # Diagonalize it
        (energy_levels, eigenstates) = linalg.eigh(H)
        # The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
        # Thus, sort them
        idx = np.argsort(energy_levels)
        # energy_levels = energy_levels[idx]
        eigenstates = eigenstates[:, idx]
        unfolded = unfold_spectrum(energy_levels)[1]
        p = distribution(unfolded, "GUE")
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
        plt.legend()
        plt.show()

    else:
        magnetic = np.linspace(0.0, 0.1, 2)

        plt.figure()
        for magneticfield in magnetic:
            H = generate_hamiltonian(L, W, magneticfield=magneticfield)
            (energy_levels, eigenstates) = linalg.eigh(H)
            # The diagonalization routine does not sort the eigenvalues (which is stupid, by the way)
            # Thus, sort them
            idx = np.argsort(energy_levels)
            # energy_levels = energy_levels[idx]
            eigenstates = eigenstates[:, idx]
            unfolded = unfold_spectrum(energy_levels)[1]
            num_bins = FreedmanDiaconis(unfolded)
            plt.hist(
                unfolded,
                bins=num_bins,
                density=True,
                histtype="step",
                color=np.random.randint(0, 255, 3) / 255,
                label=r"ULSD for $\varphi = $" f"{round(magneticfield, 3)}",
            )
        p = distribution(unfolded, "GUE")
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            p,
            "--",
            color=np.random.randint(0, 255, 3) / 255,
            label="GUE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GOE"),
            "--",
            label="GOE",
            color=np.random.randint(0, 255, 3) / 255,
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
    # GenerateDisorderSpinOrbit(L, W)
