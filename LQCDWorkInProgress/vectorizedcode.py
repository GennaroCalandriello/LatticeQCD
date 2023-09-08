import numpy as np
from numba import njit

from module.functions import *

L = 5
N = L ** 4

chi = np.zeros((N, 4), dtype=complex)
gauge = np.zeros((N, 4, 2, 2), dtype=complex)


@njit()
def GaugeChiInitialization(chi, gauge):
    """Remember that this is vectorized on L**4 lattice. Chi is the fermion spinor 4-component field"""
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    index = x + L * (y + L * (z + L * t))
                    for spin in range(4):
                        chi[index][spin] = complex(np.random.rand(), np.random.rand())
                    for mu in range(4):
                        gauge[index][mu] = SU2SingleMatrix()

    return (chi, gauge)


@njit()
def GaugeChiInitializationTwoFlavors(chi, gauge):
    """Produce 2-flavor fermions. """
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    index = x + L * (y + L * (z + L * t))
                    for spin in range(4):
                        chi[index][spin][0] = complex(
                            np.random.rand(), np.random.rand()
                        )
                        chi[index][spin][1] = complex(
                            np.random.rand(), np.random.rand()
                        )
                    for mu in range(4):
                        gauge[index][mu] = SU2SingleMatrix()

    return chi, gauge


@njit()
def StaggeredField(x, y, z, t, chi):

    index = x + L * (y + L * (z + L * t))
    psi = (-1) ** (x + y + z + t) * chi[index]

    return psi


@njit()
def FermionMatrix(M, gauge, chi):

    for i in range(N):
        for mu in range(4):
            for a in range(2):
                for b in range(2):
                    idx = i * 4 * 2
                    M[idx + mu * 2 + a, idx + mu * 2 + b] = (
                        gauge[i, mu, a, b] * chi[i, mu, a]
                    )
                    M[idx + mu * 2 + a, (i + mu) % N * 4 * 2 + mu * 2 + b] = (
                        gauge[i, mu, a, b] * chi[i, mu, a]
                    )

    return M


chi2flavor = np.zeros(
    (N, 4, 2), dtype=complex
)  # 4-dimensional spinor field, 2-dimensional flavor field
M = np.zeros((N * 4 * 2, N * 4 * 2), dtype=complex)  # define the fermion matrix
chi, gauge = GaugeChiInitializationTwoFlavors(chi2flavor, gauge)
M = FermionMatrix(M, gauge, chi)
print(M)

# import numpy as np

# N = 128 # number of lattice sites

# # Define the fermion fields
# psi = np.random.rand(N, 4, 2) # 4-dimensional spinor field, 2-dimensional flavor field
# chi = np.zeros((N, 4, 2))

# # Define the gauge fields
# gauge = np.random.rand(N, 4, 2, 2) # 4-dimensional gauge field, 2x2 matrix for SU(2)

# # Define the fermion matrix
# M = np.zeros((N*4*2, N*4*2)) # initialize the matrix

# # Fill in the matrix elements
# for i in range(N):
#     for mu in range(4):
#         for a in range(2):
#             for b in range(2):
#                 M[i*4*2 + mu*2 + a, i*4*2 + mu*2 + b] = gauge[i, mu, a, b] * psi[i, mu, a]
#                 M[i*4*2 + mu*2 + a, (i+mu)%N*4*2 + mu*2 + b] = gauge[i, mu, a, b] * chi[i, mu, b]

# # The fermion matrix is now defined and can be used for various operations, such as matrix-vector multiplications, eigenvalue computations, etc.
