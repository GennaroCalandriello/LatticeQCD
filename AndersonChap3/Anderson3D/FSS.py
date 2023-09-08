import numpy as np
from numba import njit
from sympy import plot
from andersonsymplectic import *
from module.functions import *
import matplotlib.pyplot as plt
from andersonGOEGUE import *
from mayavi import mlab
import mayavi
from scipy.sparse import linalg

# from skimage.measure import marching_cubes_lewiner
from scipy.sparse.linalg import eigsh
from skimage.measure import marching_cubes


@njit()
def Rparameter(LL):
    """-----------------Rpar-----------------
    Generates the matrices to construct R(ij)
    --------------------------------------
    njit() test: passed"""
    # L = 2 * L * L
    betaij = np.zeros((2 * LL * LL * LL, 2 * LL**3), dtype=np.complex128)
    alphaij = betaij.copy()
    gammaij = betaij.copy()

    metropolis = True
    if metropolis:
        metro = metropolisHastings()
        metro = metro[40000:]  # delete thermalization
        metro = shuffle_array(metro)

        for i in range(2 * LL**3):
            for j in range(2 * LL**3):
                betaij[i, j] = metro[i * LL * LL + j * LL]

    for i in range(2 * LL**3):
        for j in range(2 * LL**3):
            alphaij[i, j] = np.random.uniform(0, 2 * np.pi)
            gammaij[i, j] = np.random.uniform(0, 2 * np.pi)

    return betaij, alphaij, gammaij


@njit()
def Rmatrix(betaij, alphaij, gammaij):
    """-----------------R-------------------------------------------
    Generates the R(ij) matrix
    ----------------------------------------------------------------
    njit() test: passed----------------------------------------------
    Rij is symplectic, verified with is_symplectic()-----------------"""
    dir = "x"
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
    Rij[0, 0] = c00 * np.exp(1j * alphaij * np.cos(betaij))
    Rij[0, 1] = c01 * np.exp(1j * gammaij * np.sin(betaij))
    Rij[1, 0] = c10 * (-np.exp(-1j * gammaij * np.sin(betaij)))
    Rij[1, 1] = c11 * np.exp(-1j * alphaij * np.cos(betaij))

    return Rij


@njit()
def Hamiltonian3D(LL, WW):
    """-----------------Hamiltonian----------------------------------------
    Generates the Hamiltonian of the system

    njit() test: passed
    this Hamiltonian is symplectic!!! Yes bitch!
    reference: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.91.184206

    -----------------------------------------------------------------------"""
    H = np.zeros(
        (2 * LL * LL * LL, 2 * LL * LL * LL), dtype=np.complex128
    )  # Initialize Hamiltonian

    Pauli = [sigma_x, sigma_y, sigma_z]
    betaij, alphaij, gammaij = Rparameter(LL)

    # Nearest neighbors
    for i in range(LL):
        for j in range(LL):
            for k in range(LL):
                # neighbors = [
                #     (i + 1, j, k),
                #     (i, j + 1, k),
                #     (i, j, k + 1),
                # ]
                neighbors = [
                    (i + 1, j, k),
                    (i - 1, j, k),
                    (i, j + 1, k),
                    (i, j - 1, k),
                    (i, j, k + 1),
                    (i, j, k - 1),
                ]
                neighbors = [
                    (n_i % LL, n_j % LL, n_k % LL) for n_i, n_j, n_k in neighbors
                ]

                for nx, ny, nz in neighbors:
                    idx1 = 2 * (i * LL * LL + j * LL + k)
                    idx2 = 2 * (nx * LL * LL + ny * LL + nz)

                    H[idx1 : idx1 + 2, idx2 : idx2 + 2] = Rmatrix(
                        betaij[idx1, idx2],
                        alphaij[idx1, idx2],
                        gammaij[idx1, idx2],
                    )
    # On-diagonal disorder term
    for i in range(LL):
        for j in range(LL):
            for k in range(LL):
                idx = 2 * (
                    i * LL * LL + j * LL + k
                )  # Calculate the index in the Hamiltonian
                # Random disorder term
                disorder = np.random.uniform(-WW / 2, WW / 2)
                disorderMatrix = np.array(
                    [[disorder, 0], [0, disorder]], dtype=np.complex128
                )
                H[idx : idx + 2, idx : idx + 2] = disorderMatrix  # on diagonal correct

    return H


def compute_box_probabilities(eigenvectors, boxes):
    """"""  # assume eigenvectors is a numpy array of shape (2000, 2000)
    # where each column is an eigenvector

    # assume boxes is a list of lists, where each inner list gives
    # the indices of the sites within a particular box
    # For example, boxes might look like [[0, 1, 2], [3, 4, 5], ..., [1998, 1999]]
    # create an empty list to store the box probabilities"""

    mu_k = []

    # iterate over each eigenvector (eigenstate)
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]

        # compute the probability distribution across the lattice for this eigenstate
        probabilities = np.abs(eigenvector) ** 2

        # compute the total probability within each box for this eigenstate
        box_probabilities = [np.sum(probabilities[box]) for box in boxes]

        # add the list of box probabilities for this eigenstate to mu_k
        mu_k.append(box_probabilities)

    return mu_k


def localization_length22(psi, lattice):
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


def compute_localization_length(system_length, num_sites, disorder_strength, energy):
    """
    Take a look to https://arxiv.org/pdf/1004.0285.pdf
    Calculate the localization length using the transfer matrix method.
    """

    # Generate a random potential in each site, which represents the disorder
    disorder_potential = (
        np.random.rand(num_sites, system_length + 1) - 0.5
    ) * disorder_strength

    # Prepare the arrays that will be used in the calculations
    transfer_matrix_values = np.zeros(system_length + 1)
    green_function_current = 1.0 / np.sqrt(num_sites) * np.identity(num_sites)
    green_function_previous = np.zeros((num_sites, num_sites))
    temp_matrix = np.zeros((num_sites, num_sites))

    log_sum = 0.0

    # Iterate over the system
    for i in range(system_length + 1):
        # Negate the Green's function from the previous step
        green_function_previous = -green_function_previous

        # Adjust the diagonal elements to include the energy and the disorder potential
        for j in range(num_sites):
            green_function_previous[j, j] += energy - disorder_potential[j, i]
            green_function_previous[j, (j + num_sites - 1) % num_sites] -= 1.0
            green_function_previous[j, (j + 1) % num_sites] -= 1.0

        # Invert the Green's function
        green_function_previous = np.linalg.inv(green_function_previous)

        # Multiply the Green's functions
        temp_matrix = np.dot(green_function_current, green_function_previous)

        # Calculate the norm and update the log sum
        norm_factor = np.linalg.norm(temp_matrix)
        log_sum += 2.0 * np.log(norm_factor)

        # Normalize the Green's function
        green_function_current = (1.0 / norm_factor) * temp_matrix

        # Store the cumulative sum in the array
        transfer_matrix_values[i] = log_sum

    # Calculate the localization length
    localization_length = system_length / (
        transfer_matrix_values[0] - transfer_matrix_values[system_length]
    )

    # Returns an estimate of the localization length and the log of the transmission vs position
    return localization_length, transfer_matrix_values


# @njit()
def FSS(Lrange, Wrange):
    """-----------------FSS-------------------------------------------"""
    XiArray = np.zeros((len(Lrange), len(Wrange)), dtype=np.complex128)

    for l in Lrange:
        print(f"Execution for L = {l}")
        for w in Wrange:
            H = Hamiltonian3D(l, w)
            eigvals, eigvecs = linalg.eigh(H)
            XiArray[l, w] = localization_length(eigvecs, l)

    return XiArray


def advanced_mayaview(eigenstates, k, plot_type="volume", contrast_vals=[0.1, 0.25]):
    """plot_type: "volume", is the one that works bettaaah"""

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1200, 1200))
    psi = eigenstates[:, k]

    if plot_type == "volume":
        magnification = 100
        print("Max value of psi: ", np.amax(np.abs(psi)))
        x, y, z = np.mgrid[
            -magnification : magnification : L * 1j,
            -magnification : magnification : L * 1j,
            -magnification : magnification : L * 1j,
        ]

        # mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        mlab.clf()
        psi = np.abs(psi) ** 2
        src = mlab.pipeline.scalar_field(
            x, y, z, psi.reshape(L, L, L), vmin=contrast_vals[0], vmax=contrast_vals[1]
        )

        # mlab.pipeline.surface(src, opacity=0.4)
        mlab.pipeline.volume(src)
        # Change the color transfer function
        mlab.outline()
        # mlab.axes(
        #     xlabel="x [Å]",
        #     ylabel="y [Å]",
        #     zlabel="z [Å]",
        #     nb_labels=6,
        #     ranges=(-L, L, -L, L, -L, L),
        # )
        # azimuth angle
        # φ = 30
        # mlab.view(azimuth=φ)
        mlab.show()

    # if plot_type == "abs-volume":
    #     abs_max = np.amax(np.abs(eigenstates))
    #     psi = -np.log((psi) / (abs_max))

    #     vol = mlab.pipeline.volume(
    #         mlab.pipeline.scalar_field(np.abs(psi)),
    #         vmin=contrast_vals[0],
    #         vmax=contrast_vals[1],
    #     )
    #     # Change the color transfer function

    #     mlab.outline()
    #     mlab.axes(
    #         xlabel="x [Å]",
    #         ylabel="y [Å]",
    #         zlabel="z [Å]",
    #         nb_labels=6,
    #         ranges=(-L, L, -L, L, -L, L),
    #     )
    #     # azimuth angle
    #     φ = 30
    #     mlab.view(azimuth=φ, distance=N * 3.5)
    #     mlab.show()

    # elif plot_type == "contour":
    #     psi = eigenstates[k]
    #     isovalue = np.mean(contrast_vals)
    #     abs_max = np.amax(np.abs(eigenstates))
    #     psi = -np.log((psi) / (abs_max))

    #     field = mlab.pipeline.scalar_field(np.abs(psi))

    #     arr = mlab.screenshot(antialiased=False)

    #     mlab.outline()
    #     mlab.axes(
    #         xlabel="x [Å]",
    #         ylabel="y [Å]",
    #         zlabel="z [Å]",
    #         nb_labels=6,
    #         ranges=(-L, L, -L, L, -L, L),
    #     )
    #     colour_data = np.angle(psi.T.ravel()) % (2 * np.pi)
    #     field.image_data.point_data.add_array(colour_data)
    #     field.image_data.point_data.get_array(1).name = "phase"
    #     field.update()
    #     field2 = mlab.pipeline.set_active_attribute(field, point_scalars="scalar")
    #     contour = mlab.pipeline.contour(field2)
    #     contour.filter.contours = [
    #         isovalue,
    #     ]
    #     contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars="phase")
    #     s = mlab.pipeline.surface(contour, colormap="hsv", vmin=0.0, vmax=2.0 * np.pi)

    #     s.scene.light_manager.light_mode = "vtk"
    #     s.actor.property.interpolation = "phong"

    #     # azimuth angle
    #     φ = 30
    #     mlab.view(azimuth=φ, distance=N * 3.5)

    #     mlab.show()


if __name__ == "__main__":
    evects1 = np.loadtxt(
        "results/GOE/eigenstates/eigenvalues_W_20.0.txt", dtype=np.complex128
    )
    evects2 = np.loadtxt(
        "results/GOE/eigenstates/eigenvalues_W_4.0.txt", dtype=np.complex128
    )
    evects3 = np.loadtxt(
        "results/GOE/eigenstates/eigenvalues_W_17.2.txt", dtype=np.complex128
    )
    num_states = [10, 30, 100, 300, 1000, 3000]
    for i in num_states:
        print("W = 4")
        advanced_mayaview(evects2, i, plot_type="volume")
        print("W = 17.2")
        advanced_mayaview(evects3, i, plot_type="volume")
        print("W = 20")
        advanced_mayaview(evects1, i, plot_type="volume")

    pass
