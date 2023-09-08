import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as lnlg
from numba import njit
from module.functions import *
from andersonsymplectic import *

import shutil
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def Hamiltonian2D(L, W, magneticfield):
    """Generate the Hamiltonian matrix in TWO DIMENSIONS for one realization of the random disorder. The value
    of magneticfield is the flux of B added as a phase factor to the hopping terms.
    The presence of B switches the ULSD from GOE to GUE."""

    H = np.zeros((L * L, L * L), dtype=complex)
    for i in range(L):
        for j in range(L):
            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
            ]
            neighbors = [(n_i % L, n_j % L) for n_i, n_j in neighbors]
            expo = np.exp(
                -2 * 1j * np.pi * magneticfield
            )  # add a phase factor depending on the magnetic flux value
            H[i * L + j, i * L + j] = (
                np.random.uniform(-W / 2, W / 2) * expo
            )  # diagonal elements H[i * L + j, i * L +
            for n_i, n_j in neighbors:
                H[i * L + j, n_i * L + n_j] = -1
                H[n_i * L + n_j, i * L + j] = -1
    return H


@njit()
def Hamiltonian3DGOEGUE(Ws, LL, magneticfield):
    """-----------------Hamiltonian----------------------------------------
    Generates the Hamiltonian for a THREE DIMENSIONAL system with periodic boundary conditions
    For the GOE case set the magneticfield to 0.0
    For the GUE case set the magneticfield to another value

    njit() test: passed

    -----------------------------------------------------------------------"""

    H = np.zeros(
        (LL * LL * LL, LL * LL * LL), dtype=np.complex128
    )  # Initialize Hamiltonian
    # On-site disorder term

    for i in range(LL):
        for j in range(LL):
            for k in range(LL):
                idx = i * LL * LL + j * LL + k  # Calculate the index in the Hamiltonian
                # Random disorder term
                disorder = np.random.uniform(-Ws / 2, Ws / 2)
                H[idx, idx] = disorder  # on diagonal correct

    # Nearest neighbors
    for i in range(LL):
        for j in range(LL):
            for k in range(LL):
                exp_phi = np.exp(
                    -2 * 1j * np.pi * magneticfield
                )  # add a phase factor depending on the magnetic flux value
                neighbors = [
                    (i - 1, j, k),
                    (i + 1, j, k),
                    (i, j - 1, k),
                    (i, j + 1, k),
                    (i, j, k - 1),
                    (i, j, k + 1),
                ]
                neighbors = [
                    (n_i % LL, n_j % LL, n_k % LL) for n_i, n_j, n_k in neighbors
                ]

                count = 0
                for nx, ny, nz in neighbors:
                    count += 1
                    # if (
                    #     count % 1 == 0
                    #     or count % 2 == 0
                    #     or count % 3 == 0
                    #     or count % 4 == 0
                    # ):
                    #     exp_phi = 1

                    idx1 = i * LL * LL + j * LL + k
                    idx2 = nx * LL * LL + ny * LL + nz

                    H[idx1, idx2] = -1 * exp_phi
    return H


def main():
    variousB = False

    if not variousB:
        # H = Hamiltonian2D(L, W, magneticfield=0.0)
        H = Hamiltonian3DGOEGUE(W, L, magneticfield=0.0)
        print("Hamiltonian generated", H.shape)

        # Diagonalize it
        (energy_levels, eigenstates) = linalg.eigh(H)
        np.savetxt("results/energy_levelsGOE.txt", energy_levels)
        np.savetxt("results/eigenstatesGOE.txt", eigenstates)

        unfolded = unfold_spectrum(energy_levels, kind="diff")[1]

        p = distribution(unfolded, "GOE")
        plt.title("ULSD for unitary Anderson Hamiltonian", fontsize=16)
        plt.xlabel("s", fontsize=12)
        plt.ylabel(r"p(s)", fontsize=12)
        plt.legend()
        plt.hist(
            unfolded,
            bins=FreedmanDiaconis(unfolded),
            density=True,
            color="deepskyblue",
            histtype="step",
            label=f"unfolded spectrum L={L}",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GSE"),
            color="lightgray",
            linestyle="--",
            label="GSE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GUE"),
            color="deepskyblue",
            linestyle="--",
            label="GUE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GOE"),
            color="grey",
            linestyle="--",
            label="GOE",
        )
        plt.legend()
        plt.show()

    else:
        magnetic = np.linspace(0.0, 0.1, 2)

        plt.figure()
        for magneticfield in magnetic:
            H = Hamiltonian2D(L, W, magneticfield=magneticfield)
            (energy_levels, eigenstates) = linalg.eigh(H)

            unfolded = unfold_spectrum(energy_levels, kind="diff")[1]

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
            "lightgray--",
            color=np.random.randint(0, 255, 3) / 255,
            label="GUE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GSE"),
            "lightgray--",
            color=np.random.randint(0, 255, 3) / 255,
            label="GSE",
        )
        plt.plot(
            np.linspace(min(unfolded), max(unfolded), len(p)),
            distribution(unfolded, "GOE"),
            "b--",
            label="GOE",
            color=np.random.randint(0, 255, 3) / 255,
        )
        plt.legend()
        plt.show()


def mainFSS(kind):
    """function divided into two parts:
    1. Generate the data
    2. Execute the MFSS analysis
    """

    # Larr = np.array([4, 6, 8, 10, 12, 14, 16])
    Larr = np.array([15])

    # 1. Generate the data
    if kind == "GOE":
        magneticfield = 0.0
        Warr = np.linspace(15.0, 20, 30)
    if kind == "GUE":
        # GUE
        magneticfield = 0.4
        # Warr = np.linspace(4.0, 20, 30)
        Warr = np.array([4.0, 6.0, 12.222, 15.111, 18.111, 20, 22])

    if kind == "GOE" or kind == "GUE":
        for LL in Larr:
            path1 = f"results/{kind}/{LL}/eigenvalues"
            path2 = f"results/{kind}/{LL}/eigenstates"

            if os.path.exists(path1) and os.path.exists(path2):
                shutil.rmtree(path1)
                shutil.rmtree(path2)
            else:
                os.makedirs(path1)
                os.makedirs(path2)

            for Ws in Warr:
                print("Generating for Ws = ", Ws)
                H = Hamiltonian3DGOEGUE(Ws, LL, magneticfield=magneticfield)
                print("Hamiltonian generated, diagonalizing...", H.shape)
                (energy_levels, eigenstates) = linalg.eigh(H)

                np.savetxt(f"{path1}/eigenvalues_W_{round(Ws)}.txt", energy_levels)
                np.savetxt(f"{path2}/eigenstates_W_{round(Ws)}.txt", eigenstates)
                print("Saved")


def rename_files_in_folder(directory):
    """
    Rename files in the specified directory that match the pattern "eigenvalues_W_<number>.txt".
    The number is rounded to two decimal places.

    Args:
    - directory (str): The directory containing the files to rename. Defaults to the current directory.
    """
    for filename in os.listdir(directory):
        if "eigenvalues_W_" in filename or "eigenstates_W_" in filename:
            parts = filename.split("_")
            number_part = parts[2].replace(".txt", "")

            # Round the number to two decimal places
            rounded_number = round(float(number_part), 1)

            # Construct the new filename
            new_filename = f"eigenvalues_W_{rounded_number}.txt"

            # Rename the file
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_filename)
            )


# Example of using the function:


if __name__ == "__main__":
    # mainFSS("GSE")
    ev1 = np.loadtxt("results/GUE/20/eigenvalues/eigenvalues_W_4.txt")
    ev2 = np.loadtxt("results/GUE/20/eigenvalues/eigenvalues_W_15.txt")
    ev3 = np.loadtxt("results/GUE/20/eigenvalues/eigenvalues_W_18.txt")
    ev4 = np.loadtxt("results/GUE/20/eigenvalues/eigenvalues_W_22.txt")

    unfolded1 = unfold_spectrum(ev1, kind="diff")[1]
    unfolded2 = unfold_spectrum(ev2, kind="diff")[1]
    unfolded3 = unfold_spectrum(ev3, kind="diff")[1]
    unfolded4 = unfold_spectrum(ev4, kind="diff")[1]

    plotting(unfolded1, "GUE", 4.0)
    plotting(unfolded2, "GUE", 15)
    plotting(unfolded3, "GUE", 18)
    plotting(unfolded4, "GUE", 22)
