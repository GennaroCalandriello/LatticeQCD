import numpy as np

# # Define the gamma matrices
gamma_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma_1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
gamma_2 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [1, 0, 0, 0]])
gamma_3 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])

gamma = np.zeros(((4, 4, 4)), dtype=complex)

gamma[0] = gamma_0
gamma[1] = gamma_1
gamma[2] = gamma_2
gamma[3] = gamma_3


# Define the Pauli matrix basis
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])


def BispinorSigma():

    """Calculate the 6 independent components of s_munu=-i/4([g_mu*g_nu-g_nu*g_mu])
    reference: https://en.wikipedia.org/wiki/Bispinor"""

    sigma = np.zeros((12, 4, 4), complex)

    i = 0
    for mu in range(4):
        for nu in range(4):
            sigma_munu = (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]) * (-1j / 4)
            if mu != nu:
                sigma[i] = sigma_munu
                i += 1

    for k in range(12):
        eigs, _ = np.linalg.eig(sigma[k])
        for e in eigs:
            if e.real == 0:
                sigma[k] = 0

    idxlist = []
    numlist = np.arange(0, 12, 1).tolist()

    for k in range(12):
        if sigma[k].any() == 0:
            idxlist.append(k)

    sigmaLI = np.zeros((6, 4, 4), complex)

    for idx in idxlist:
        numlist.remove(idx)

    m = 0
    for idx in numlist:
        sigmaLI[m] = sigma[idx]
        m += 1

    return sigmaLI


sigma = BispinorSigma()


