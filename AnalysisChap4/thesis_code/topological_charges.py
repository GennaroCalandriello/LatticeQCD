import numpy as np
from main import *

def topological_charge():
    d, c, topodec, topoconf = loadtxt(topocool=True)

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
    charges =topo[:, 2]
    matrix = charges.reshape(-1, 16)
    print(matrix)

    #need squared figure
    plt.figure(figsize=(10, 10))
    # Plot the first 10 configurations
    for i in range(10):
        plt.plot(range(i*16, (i+1)*16), matrix[i], label=f"Config {i+1}", marker='x', linestyle='None', markersize=7)

    # Add vertical lines to indicate configurations
    for i in range(0, 160, 16):
        plt.axvline(x=i, color='grey', linestyle='--')

    # Set x-ticks to indicate configuration number
    ticks = [i*16 + 7 for i in range(10)]  # Centers the configuration numbers between the vertical lines
    labels = [str(i+1) for i in range(10)]
    plt.xticks(ticks, labels)

    # Add horizontal grid lines for integer values of topological charges
    y_min = int(min(charges))
    y_max = int(max(charges))
    for y in range(y_min, y_max + 1):
        plt.axhline(y=y, color='lightgray', linestyle='--')

    plt.grid(axis='y', which='major', linestyle='--', linewidth=0.5)

    plt.legend()
    plt.xlabel("Configuration", fontsize=15)
    plt.ylabel("Topological Charges", fontsize=15)
    plt.title(f"Topological Charges during cooling, {phase} phase", fontsize=16)
    plt.tight_layout()
    plt.show()
    

def minimization():
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
        deviation = scaled_QL - np.round(alpha*scaled_QL)
        # Calculate the mean squared deviation
        mean_squared_deviation = np.mean(deviation ** 2)

        return mean_squared_deviation

    # Perform the minimization of the objective function
    result = minimize_scalar(objective, bounds=(0.1, 2), args=(QL_array,), method='bounded')

    # The optimal alpha value
    alpha_optimal = result.x
    print(f"The optimal alpha value is: {alpha_optimal}")

    # Calculate the optimized topological charge array
    Q_optimized = np.round(alpha_optimal * QL_array)

    print(f'Chi original for {phase}', susceptibility(QL_array))
    print(f'Chi optimized for {phase}',susceptibility(Q_optimized))
    print("Error on chi optimized", susceptibility_error(Q_optimized))
    

    

def susceptibility(Q):
    chi = np.mean(Q**2)/(Volume*Nt)
    return chi*10**5

def susceptibility_error(Q):
    sigma = errorAnalysis(None, Q, kind = 4)
    return sigma

if __name__ == "__main__":
    # topological_charge()
    minimization()
    pass