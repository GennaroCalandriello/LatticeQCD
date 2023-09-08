import numpy as np
import matplotlib.pyplot as plt

eigenstates = np.loadtxt("eigenvectors_mod_Sinai.txt", dtype=complex)


Nx, Ny = 300, 300


def oneD_to_twoD(Nx, Ny, psi):

    """Transform vectors in matrices"""

    count = 0
    PSI = np.zeros([Nx, Ny], dtype="complex")
    for i in range(Nx):
        for j in range(Ny):
            PSI[i, j] = psi[count]
            count += 1
    return PSI


for n in range(20):
    psi = eigenstates[:, n]
    PSI = oneD_to_twoD(Nx, Ny, psi)
    PSI = np.abs(PSI) ** 2
    plt.pcolormesh(np.flipud(PSI), cmap="terrain")
    plt.axis("equal")
    plt.axis("off")
    plt.show()
