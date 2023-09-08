import numpy as np
from main import *
import matplotlib.pyplot as plt

edeconfined, econfined, topodeconfinato, topoconfinato = loadtxt(
    newresults=True, topocool=True
)

print("Deconfined", topodeconfinato.shape)
plt.plot(topodeconfinato[0:200, 1], "o")
plt.show()
