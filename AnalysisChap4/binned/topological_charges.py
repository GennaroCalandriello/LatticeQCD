import numpy as np
from main import *


def topological_charge():
    d, c, topodec, topoconf = loadtxt(topocool=True)
    numconf = 20
    initial_conf = 0

    # caso deconfinato: sono 16 cariche topologiche per ogni configurazione (5312/332 = 16)
    if phase == "confined":
        topo = topoconf
    elif phase == "deconfined":
        topo = topodec
    print(topo.shape)
    topotot = []
    print(len(topo[:, 2]))
    # averages = []
    # _ = CDF(topo[:, 2])

    print(topo[:, 2])
    charges = topo[:, 2]
    matrix = charges.reshape(-1, 16)
    print(matrix)

    # need squared figure
    plt.figure(figsize=(12, 7))
    # Plot the first numconf configurations
    for i in range(numconf):
        plt.plot(
            range(i * 16, (i + 1) * 16),
            matrix[initial_conf + i],
            label=f"Config {initial_conf+i+1}",
            marker="x",
            linestyle="None",
            markersize=7,
        )

    # Add vertical lines to indicate configurations
    for i in range(0, 16 * numconf + 1, 16):
        plt.axvline(x=i, color="grey", linestyle="--", linewidth=0.5)

    # Set x-ticks to indicate configuration number
    ticks = [i for i in range(7, numconf * 16 + 1, 5 * 16)]

    labels = [str(i // 16 + 1) for i in ticks]
    plt.xticks(ticks, labels, fontsize=12)

    # Add horizontal grid lines for integer values of topological charges
    y_min = int(-20)
    y_max = int(20)

    for y in range(y_min, y_max + 1):
        plt.axhline(y=y, color="lightgray", linestyle="--")

    plt.grid(axis="y", which="major", linestyle="--", linewidth=0.5)

    # plt.legend()
    plt.xlabel("Configuration", fontsize=15)
    plt.ylabel("Topological Charges", fontsize=15)
    plt.title(f"Topological Charges during cooling, {phase} phase", fontsize=16)
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.show()


def minimization(phase):
    from scipy.optimize import minimize_scalar

    d, c, topodec, topoconf = loadtxt(topocool=True)

    # caso deconfinato: sono 16 cariche topologiche per ogni configurazione (5312/332 = 16)
    if phase == "confined":
        topo = topoconf
    elif phase == "deconfined":
        topo = topodec
    # topo = topo[:, 2]

    QL_array = []

    for i in range(len(topo)):
        if topo[i, 1] == 150:
            QL_array.append(topo[i, 2])

    QL_array = np.array(QL_array)

    def objective(alpha, QL_array):
        # Calculate the scaled charges
        scaled_QL = alpha * QL_array
        # Calculate the deviation from the nearest integer
        deviation = scaled_QL - np.round(alpha * scaled_QL)
        # Calculate the mean squared deviation
        mean_squared_deviation = np.mean(deviation**2)

        return mean_squared_deviation

    # Perform the minimization of the objective function
    result = minimize_scalar(
        objective, bounds=(0.7, 1.5), args=(QL_array,), method="bounded"
    )

    # The optimal alpha value
    alpha_optimal = result.x
    print(f"The optimal alpha value is: {alpha_optimal}")

    # Calculate the optimized topological charge array
    Q_optimized = np.round(alpha_optimal * QL_array)

    print(f"Chi original for {phase}", susceptibility(QL_array))
    print(f"Chi optimized for {phase}", susceptibility(Q_optimized))
    print("Error on chi optimized", susceptibility_error(Q_optimized))

    return QL_array, Q_optimized


def both_phases_behavior():
    # now plot the topological charge across configurations
    QL_arr_dec, Q_optimized_dec = minimization("deconfined")
    QL_arr_conf, Q_optimized_conf = minimization("confined")
    if phase == "deconfined":
        QL_array = QL_arr_dec
        Q_optimized = Q_optimized_dec
    elif phase == "confined":
        QL_array = QL_arr_conf
        Q_optimized = Q_optimized_conf

    plt.figure(figsize=(8, 7))
    # now the label each 30 configurations
    plt.xlabel("Configuration", fontsize=15)
    plt.ylabel("Topological Charge", fontsize=15)
    plt.title(f"Topological Charge for {phase} phase", fontsize=16)
    plt.scatter(
        range(len(QL_array)), QL_array, label="Original", color="red", marker="x"
    )
    plt.scatter(
        range(len(Q_optimized)),
        Q_optimized,
        label="Optimized",
        color="darkorchid",
        marker="+",
    )
    plt.legend()
    plt.grid(axis="y", which="major", linestyle="--", linewidth=0.5)
    plt.show()

    # now plot the distribution of the topological charge
    plt.figure(figsize=(8, 7))
    plt.xlabel("Q", fontsize=15)
    plt.ylabel("P(Q)", fontsize=15)
    plt.title(f"P(Q) for both phases", fontsize=16)
    # istogramma tratteggiato

    plt.hist(
        Q_optimized_dec,
        bins=FreedmanDiaconis(Q_optimized_dec),
        label="Deconfined",
        histtype="step",
        density=True,
        color="red",
        linestyle="--",
        linewidth=0.8,
    )
    plt.hist(
        Q_optimized_conf,
        bins=FreedmanDiaconis(Q_optimized_conf),
        label="Confined",
        histtype="step",
        density=True,
        color="darkorchid",
        linestyle="-.",
        linewidth=0.8,
    )
    plt.legend()
    plt.grid()
    plt.show()

    # total number of dinstinct topological objects:
    N_II_conf = np.mean(Q_optimized_conf**2)
    N_II_dec = np.mean(Q_optimized_dec**2)
    print(f"Number of distinct topological objects in conf phase: {N_II_conf}")
    print(f"Number of distinct topological objects in decaffeinata phase: {N_II_dec}")


def susceptibility(Q):
    chi = np.mean(Q**2) / (Volume * Nt)
    return chi * 10**5


def susceptibility_error(Q):
    sigma = errorAnalysis(None, Q, kind=4)
    return sigma / np.sqrt(len(Q) - 1)


if __name__ == "__main__":
    # topological_charge()
    # minimization()
    both_phases_behavior()
    pass
