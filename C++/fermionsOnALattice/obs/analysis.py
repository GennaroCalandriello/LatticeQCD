import numpy as np
import matplotlib.pyplot as plt

wilson = np.loadtxt("../func/Wilson11.txt")
topological = np.loadtxt("../func/TopologicalCharge.txt")
Nstep = len(wilson)

plt.figure(figsize=(8, 7))
plt.plot(
    np.arange(Nstep),
    1 - wilson,
    label="Wilson",
    color="blue",
    marker="+",
    linestyle="--",
    linewidth=1,
)
plt.title("Wilson, Cooling", fontsize=15)
plt.xlabel("Nstep", fontsize=15)
plt.ylabel(r"$\langle W \rangle$", fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 7))
# marker list
markers = ["+", "o", "*", "s", "x", "D", "^", "v", ">", "<", "p"]
line_styles = ["--"] * len(markers)
# plot all files starting with "TopologicalCharge"
for i in range(0, 4):
    topological = np.loadtxt(f"../func/TopologicalCharge{i}.txt")
    plt.plot(
        np.arange(Nstep),
        topological,
        label=f"Configuration {i+1}",
        marker=markers[i],
        linestyle=line_styles[i],
        linewidth=1,
    )
# plt.scatter(
#     np.arange(Nstep), topological, label="Topological Charge", color="red", marker="+"
# )
plt.title("Topological Charge, Cooling", fontsize=15)
plt.xlabel("Nstep", fontsize=15)
plt.ylabel("Q", fontsize=15)
# legenda fontsize
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
