import numpy as np
import matplotlib.pyplot as plt

b0 = 0.25 * (11 - 2 / 3 * 10)
mu0 = 1.0
Lambdaqcd = 0.2
# # Load data


# analytic solution of the qcd beta function
def alpha_s(mu):
    alpha = 1.0 / (1 + (1 / (2 * np.pi)) * b0 * np.log(mu / mu0))
    return alpha


def alpha_s2(mu):
    alpha = 1 / (b0 * np.log(mu / np.sqrt(Lambdaqcd)))
    return alpha


# calculate analytic result
mu_anal = np.linspace(1, 10, 1000)
alpha_analytic = alpha_s(mu_anal)
# Plot

plt.figure()
for i in range(4):
    alpha = np.loadtxt("alpha_" + str(i) + ".txt")
    mu = np.loadtxt("mu_" + str(i) + ".txt")
    if i == 0:
        plt.plot(mu, alpha, label="tree level")
    else:
        plt.plot(mu, alpha, label=str(i + 1) + " loops")
plt.plot(mu_anal, alpha_analytic, "--", label="analytic 1-loop")
plt.xlabel(r"$\mu (GeV)$", fontsize=15)
plt.ylabel(r"$\alpha_s(\mu)$", fontsize=15)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
