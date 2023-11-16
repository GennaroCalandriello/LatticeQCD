import numpy as np

from utils.gamma import *
from utils.params import *
from test_lavoroqui import *

"""References:
1. Luscher: "Implementation of the lattice Dirac operator" """


def FieldStrength(U, x, y, z, t, indexinverter):

    # Q = np.array(
    #     ((0 + 0j, 0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j, 0 + 0j))
    # )
    Q = np.zeros((4, 4, su3, su3), dtype=complex)

    for mu in range(4):
        a_mu = [0, 0, 0, 0]
        a_mu[mu] = 1
        for nu in range(4):
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            if (
                indexinverter
            ):  # sta cosa va controllata meglio, sennò conviene riscriverlo
                mu = nu
                a_nu = a_mu

            Q[mu, nu] = (
                U[x, y, z, t, mu]
                @ U[
                    (x + a_mu[0]) % Nx,
                    (y + a_mu[1]) % Ny,
                    (z + a_mu[2]) % Nz,
                    (t + a_mu[3]) % Nt,
                    nu,
                ]
                @ U[
                    (x + a_nu[0]) % Nx,
                    (y + a_nu[1]) % Ny,
                    (z + a_nu[2]) % Nz,
                    (t + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[x, y, z, t, nu].conj().T
                + U[x, y, z, t, nu]
                @ U[
                    (x - a_mu[0] + a_nu[0]) % Nx,
                    (y - a_mu[1] + a_nu[1]) % Ny,
                    (z - a_mu[2] + a_nu[2]) % Nz,
                    (t - a_mu[3] + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    mu,
                ]
                + U[
                    (x - a_mu[0]) % Nx,
                    (y - a_mu[1]) % Ny,
                    (z - a_mu[2]) % Nz,
                    (t - a_mu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0] - a_nu[0]) % Nx,
                    (y - a_mu[1] - a_nu[1]) % Ny,
                    (z - a_mu[2] - a_nu[2]) % Nz,
                    (t - a_mu[3] - a_nu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_mu[0] - a_nu[0]) % Nx,
                    (y - a_mu[1] - a_nu[1]) % Ny,
                    (z - a_mu[2] - a_nu[2]) % Nz,
                    (t - a_mu[3] - a_nu[3]) % Nt,
                    mu,
                ]
                @ U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    nu,
                ]
                + U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_nu[0]) % Nx,
                    (y - a_nu[1]) % Ny,
                    (z - a_nu[2]) % Nz,
                    (t - a_nu[3]) % Nt,
                    mu,
                ]
                @ U[
                    (x + a_mu[0] - a_nu[0]) % Nx,
                    (y + a_mu[1] - a_nu[1]) % Ny,
                    (z + a_mu[2] - a_nu[2]) % Nz,
                    (t + a_mu[3] - a_nu[3]) % Nt,
                    nu,
                ]
                @ U[x, y, z, t, mu].conj().T
            )
    return Q


def Fmunu(U, x, y, z, t):

    Qmunu = FieldStrength(U, x, y, z, t, indexinverter=False)
    Qnumu = FieldStrength(U, x, y, z, t, indexinverter=True)

    Fmunu = 1 / 8 * (Qmunu - Qnumu)

    return Fmunu


def DiracNaive(U, D, psi, x, y, z, t):

    """This returns a 4x4 matrix on each lattice point, but it should be converted in 
    a sparse 12x12 matrix to sum it with the sigmaxField term, e come si fa?"""

    for alpha in range(4):
        for beta in range(4):
            Dtemp = 0
            for a in range(3):
                for b in range(3):
                    for mu in range(4):
                        a_mu = [0, 0, 0, 0]
                        a_mu[mu]
                        Dtemp += 0.5 * (
                            U[x, y, z, t, mu, a, b]
                            * (1 - gamma[mu][alpha, beta])
                            * psi[
                                (x + a_mu[0]) % N,
                                (y + a_mu[1]) % N,
                                (z + a_mu[2]) % N,
                                (t + a_mu[3]) % N,
                                alpha,
                                a,
                            ]
                            + U[
                                (x - a_mu[0]) % N,
                                (y - a_mu[1]) % N,
                                (z - a_mu[2]) % N,
                                (t - a_mu[3]) % N,
                                mu,
                                a,
                                b,
                            ]
                            .conj()
                            .T
                            * (1 + gamma[mu][alpha, beta])
                            * psi[
                                (x - a_mu[0]) % N,
                                (y - a_mu[1]) % N,
                                (z - a_mu[2]) % N,
                                (t - a_mu[3]) % N,
                                beta,
                                b,
                            ]
                        )
            D[x, y, z, t, alpha, beta] = Dtemp

    return D[x, y, z, t]


def DiracNaive2(U, D, psi, x, y, z, t):

    """This returns a 4x4 matrix on each lattice point, but it should be converted in 
    a sparse 12x12 matrix to sum it with the sigmaxField term, e come si fa?"""

    psicoupling = True
    m = 0.2
    for alpha in range(4):
        for beta in range(4):
            Dtemp = 0
            for mu in range(4):
                a_mu = [0, 0, 0, 0]
                a_mu[mu] = 1
                if psicoupling:
                    Dtemp += 0.5 * (
                        U[x, y, z, t, mu]
                        * (1 - gamma[mu][alpha, beta])
                        * (
                            psi[
                                (x + a_mu[0]) % N,
                                (y + a_mu[1]) % N,
                                (z + a_mu[2]) % N,
                                (t + a_mu[3]) % N,
                                mu,
                            ]
                        )
                        + U[
                            (x - a_mu[0]) % N,
                            (y - a_mu[1]) % N,
                            (z - a_mu[2]) % N,
                            (t - a_mu[3]) % N,
                            mu,
                        ]
                        .conj()
                        .T
                        * (1 + gamma[mu][alpha, beta])
                        * psi[
                            (x - a_mu[0]) % N,
                            (y - a_mu[1]) % N,
                            (z - a_mu[2]) % N,
                            (t - a_mu[3]) % N,
                            mu,
                        ]
                        .conj()
                        .T
                    )
                else:
                    Dtemp += 0.5 * (
                        U[x, y, z, t, mu]
                        * (1 - gamma[mu][alpha, beta])
                        * (
                            delta4(
                                (x + a_mu[0]) % N,
                                (y + a_mu[1]) % N,
                                (z + a_mu[2]) % N,
                                (t + a_mu[3]) % N,
                                mu,
                            )
                        )
                        + U[
                            (x - a_mu[0]) % N,
                            (y - a_mu[1]) % N,
                            (z - a_mu[2]) % N,
                            (t - a_mu[3]) % N,
                            mu,
                        ]
                        .conj()
                        .T
                        * (1 + gamma[mu][alpha, beta])
                        * delta4(
                            (x - a_mu[0]) % N,
                            (y - a_mu[1]) % N,
                            (z - a_mu[2]) % N,
                            (t - a_mu[3]) % N,
                            mu,
                        )
                    )

        D[x, y, z, t, alpha, beta] = Dtemp + m

    return D[x, y, z, t]


def scalarmultiplication(matrix, k):

    for i in range(len(matrix[0])):
        for j in range(len(matrix[:, 0])):
            matrix[i, j] = matrix[i, j] * k

    return matrix


def WilsonDirac():
    return 1


def sigmaxField(sigma, Field, coupling):
    """Field is a 4x4x3x3 matrix, sigma are the 6 bispinors linearly independent generating
    the Clifford algebra for the spinor fields. The coupling is (maybe?) a sum 
    over all the 6 contributions of the sigmas LI. Clearly it is referred to each lattice site.
    The coupling is also a 4x4x3x3 matrix with of zeros and only complex values"""
    for s in range(6):
        for mu in range(4):
            for nu in range(4):
                coupling[mu, nu] += 1j * scalarmultiplication(
                    Field[mu, nu], sigma[s][mu, nu]
                )
    # for mu in range(4):
    #     for nu in range(mu, 4):  # 1234 1234
    #         if mu != nu:
    #             coupling[mu, nu] += 1j * np.kron(Field[mu, nu], sigma[mu + nu])

    return coupling


def delta(i, j):

    """Kronecker delta function"""

    return int(i == j)


def delta4(x, y, z, t, mu):

    """4 dimensional delta function as product of 3 delta functions acting
    on pairs of coupled indices"""

    return delta(x, mu) * delta(y, mu) * delta(z, mu) * delta(t, mu)


if __name__ == "__main__":

    from scipy.sparse import coo_matrix

    # declaration
    U = initializefield(np.zeros((Nx, Ny, Nt, Nz, 4, su3, su3), complex))
    # for _ in range(5):
    #     U = HB_updating_links(1.7, U, N)
    psi = np.zeros((Nx, Ny, Nz, Nt, 4, 3), complex)
    psi = quarkfield(psi, forloops=True)
    D = np.zeros((Nx, Ny, Nz, Nt, 4, 4, 3, 3), complex)
    Devenodd = np.zeros((Nx, Ny, Nz, Nt, 4, 4, 3, 3), complex)
    Field = Devenodd.copy()

    exe = False

    sigma = BispinorSigma()

    TOTAL = np.zeros((Nx, Ny, Nz, Nt, 4, 4, 3, 3), complex)
    eigsolist = []
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    coupling = np.zeros((4, 4, 3, 3), complex)
                    eigso = 0
                    Field = Fmunu(U, x, y, z, t)
                    naive = DiracNaive2(U, D, psi, x, y, z, t)
                    Field = sigmaxField(sigma, Field, coupling)
                    SW = Field
                    TOTAL[x, y, z, t] = SW

                    # eigso, _ = np.linalg.eig(SW)
                    # # eigso = spacing(eigso.flatten())
                    # eigso = np.array(eigso)
                    # eigsolist.append(eigso.flatten())

    eigsolist, _ = np.linalg.eig(TOTAL)
    eigsolist = np.array(eigsolist).flatten()
    index = np.nonzero(eigsolist)
    eigsnonzero = eigsolist[index]
    eigso = spacing(eigsnonzero)
    plt.hist(eigso.imag, bins=30, histtype="step")
    plt.show()
    # row, col = np.nonzero(naive)
    # data = naive[row, col]
    # coo = coo_matrix((data, (row, col)), shape=(12, 12))
    # print(coo)
    # eigs,_=np.linalg.eig(coo)
    # plt.hist(eigs, 50)
    # plt.show()


# ChatGPT tells me this:

# You are correct, I apologize for the confusion in my previous response.
# The field strength tensor Fμν is a 3x3 matrix (NO) in color space, and is given by:

# Fμν = ∂μAν - ∂νAμ + ig[Aμ, Aν]

# where Aμ is the gauge potential and g is the coupling constant. In order to couple
# the sigma bispinor with the field strength tensor in SU(3) gauge theory,
# you need to first contract the sigma bispinor with the gamma matrices to obtain
# the six linearly independent 4x4 matrices. Then, for each component of the field
# strength tensor, you can form a 4x4 matrix by taking the Kronecker product of the
#  3x3 matrix with the 2x2 identity matrix in Dirac space. This gives you a total of 18
# independent 4x4 matrices that you can use to construct terms in the action.

# To be more specific, let σ(μν) denote the six independent sigma bispinor matrices, as before.
#  Then, for each component of the field strength tensor Fμν, you can define a 4x4 matrix G(μν) as follows:

# G(μν) = F(μν) ⊗ I

# where ⊗ denotes the Kronecker product, F(μν) is the 3x3 matrix representing the (μ,ν) component
#  of the field strength tensor in color space, and I is the 2x2 identity matrix in Dirac space.
#  You can then construct terms in the action by contracting the sigma bispinor with the 4x4 matrices
#  G(μν) using the trace over the Dirac indices, as follows:

# S = g ∫ d⁴x Tr[σ(μν) G(μν)]

# where g is the coupling constant. This gives you a total of 18 independent terms in the action that
# couple the sigma bispinor with the field strength tensor in SU(3) gauge theory.

# si è corretto e poi ha scritto:

# I apologize for the confusion in my previous response.
# You are correct that the field strength tensor is a 4x4x3x3 matrix,
#  while the sigma bispinor is a 6x4x4 matrix. In order to couple the
#   bispinor to the field strength tensor, you will need to perform a
#    contraction over the color indices. This can be done by taking the
#     trace of the product of the sigma bispinor and the field strength tensor.
#      This will give you a 6x4x4 matrix, which can then be contracted with
#       the quark fields to form a Dirac bilinear.

# In summary, to couple the sigma bispinor with the field strength tensor
#  in SU(3) lattice QCD, you will need to compute the trace of the product
#   of the bispinor and the field strength tensor, which will result in a
#   6x4x4 matrix.

# il suo esempio di contrazione sugli indici di colore:
###########################################################################
# import numpy as np

# # Define the sigma bispinor
# sigma = np.zeros((6, 4, 4), dtype=np.complex128) #ok

# # Define the field strength tensor
# F = np.zeros((4, 4, 3, 3), dtype=np.complex128) #ok

# # Compute the trace of the product of sigma and F
# bispinor_F = np.zeros((6, 4, 4), dtype=np.complex128)
# for a in range(3): #???
#     for b in range(3): #???
#         bispinor_F += np.einsum('abc,ijklbc->ijkl', sigma[a], F[:, :, a, b]) * np.einsum('abc,ijklbc->ijkl', sigma[b], np.conjugate(F[:, :, a, b]).T)

# # The resulting matrix bispinor_F will be a 6x4x4 matrix
################################################################################
