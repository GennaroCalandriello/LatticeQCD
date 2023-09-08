import numpy as np
from numba import njit

from module.functionsu3 import *
from module.level_spacing_eval import *
from utils.gamma import *
from utils.params import *

# @njit()
def UgammaProd(gamma, U, res):
    for alpha in range(4):
        for beta in range(4):
            total = 0
            for a in range(3):
                for b in range(3):
                    total += gamma[alpha, beta] * (
                        U[a, b] * quark[alpha, a] - U[a, b] * quark[beta, b]
                    )
            res[alpha, beta] = total

    print(res)


def prova():
    x = np.random.rand(1, 8, 2, 8, 10)
    y = np.random.rand(8, 10, 10)
    z = np.einsum("nkctv,kvw->nctw", x, y)
    print(z)

    R = np.zeros((1, 2, 8, 10))

    for n in range(1):
        for c in range(2):
            for t in range(8):
                for w in range(10):
                    total = 0
                    # These are the variables to sum over
                    for v in range(10):
                        for k in range(8):
                            total += x[n, k, c, t, v] * y[k, v, w]
                    R[n, c, t, w] = total

    print(R)

    if z.any() == R.any():
        print("thats all folks")


res = np.zeros((4, 4), dtype=complex)
quark = np.random.rand(4, 3) + 1j * np.random.rand(4, 3)

############################################################################################
@njit()
def initializefield(U):

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = SU3SingleMatrix()
    return U


@njit()
def quarkfield(quark, forloops):

    """Each flavor of fermion (quark) has 4 space-time indices, 3 color
    indices and 4 spinor (Dirac) indices.
    The components of the spinor are related to the behavior of the particle in spacetime,
    and they can be interpreted as representing different aspects of the particle's wavefunction.
    For example, the first two components of the spinor are often referred to as the particle and 
    antiparticle components, respectively, while the last two components are related to the particle's
    spin.

    The precise relation between the components of the Dirac spinor is determined by the specific form
    of the gamma matrices and the physical conditions under which the particle is considered. 
    The Dirac equation is a linear partial differential equation, and the relationship between
    its components is determined by the coefficients of the equation, as well as the boundary
    conditions that apply to the problem at hand."""

    if forloops:
        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    for t in range(Nt):
                        for spinor in range(Dirac):
                            for Nc in range(color):
                                quark[x, y, z, t, spinor, Nc] = complex(
                                    np.random.rand(),
                                    np.random.rand(),  #### io questo non creto
                                )
    # oppure più banalmente
    if not forloops:
        quark = np.random.rand(Nx, Ny, Nz, Nt, Dirac, color) + 1j * np.random.rand(
            Nx, Ny, Nz, Nt, color, Dirac
        )

    return quark


def DiracMatrix(U, psi, D):

    # The Dirac matrix in lattice QCD is represented as a 4x4 complex matrix for each lattice site.

    m = 1.7
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    # D[x, y, z, t] += m
                    for alpha in range(4):
                        for beta in range(4):
                            Dtemp = 0
                            for a in range(3):
                                for b in range(3):
                                    for mu in range(4):

                                        a_mu = [0, 0, 0, 0]
                                        a_mu[mu] = 1
                                        Dtemp += 0.5 * (
                                            gamma[mu][alpha, beta]
                                            * (
                                                m
                                                + U[x, y, z, t, mu, a, b]
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
                                                    b,
                                                    a,
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
                                                .conj()
                                                .T
                                                * gamma[0][alpha, beta]
                                            )
                                        )

                            D[x, y, z, t, alpha, beta] = Dtemp
                    # D[x, y, z, t] = D[x, y, z, t] / np.linalg.norm(D[x, y, z, t])

    return D


def psibar(psi):
    psibar = psi.conj().T @ gamma[0]
    return psibar


def spacing(eigs):
    spac = []
    # eigs = np.sort(eigs)

    for i in range(1, len(eigs) - 1):
        spac.append((eigs[i + 1] - eigs[i], eigs[i] - eigs[i - 1]))

    return (spac)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    heatbath = True

    psi = quarkfield(np.zeros((Nx, Ny, Nz, Nt, 4, 3), complex), True)
    U = initializefield(np.zeros((Nx, Ny, Nz, Nt, 4, 3, 3), complex))

    if heatbath:
        for _ in range(100):
            print("heatbath numero", _)
            U = HB_updating_links(9.6, U, N)

    Dirac = np.zeros((Nx, Ny, Nz, Nt, 4, 4), complex)
    D_nm = DiracMatrix(U, psi, Dirac)

    eigs, _ = np.linalg.eig(D_nm)
    eigs = eigs.flatten()
    eigs = eigs.real
    eigspure = eigs.copy()

    print("number of eigenvalues: ", len(eigs))

    ###-------------------------------------------Unfolding--------------------------------------
    eigs = np.sort(eigs)  # first step is the sort of eigenvalues
    unfolded = unfolding_2_punto_0(eigs)  # unfold the spectrum
    spac = spacing(unfolded)
    spac = spac / np.mean(spac)
    ###-------------------------------------------------------------------------------------------

    ###----------------------------Statistical predictions from theoretical distributions---------
    GUE = distribution(spac, "GUE")
    GSE = distribution(spac, "GSE")
    GOE = distribution(spac, "GOE")
    POISSON = distribution(spac, "Poisson")
    ###--------------------------------------------------------------------------------------------
    e, _ = np.linalg.eig(D_nm[0, 0, 0, 2])
    print(e)
    x = np.linspace(0, max(spac), len(spac))
    plt.figure()
    plt.hist(
        spac,
        60,
        density=True,
        histtype="step",
        fill=False,
        color="b",
        label="Spacing distribution",
    )
    plt.plot(x, GUE, "g--", label="GSE")
    plt.plot(x, GSE, "b--", label="GSE")
    plt.plot(x, GOE, "r--", label="GOE")
    plt.plot(x, POISSON, "y--", label="Poisson")
    plt.legend()
    plt.show()

    print("mean spacing", np.mean(spac))

    ###----------------------Normal spectrum histogram-------------------------------
    plt.figure()
    plt.hist(np.sort(spacing(eigspure)), 60, density=True, histtype="step")
    plt.show()
    ###------------------------------------------------------------------------------
