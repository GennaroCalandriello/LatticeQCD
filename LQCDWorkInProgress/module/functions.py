import numpy as np
from functionsu3 import *


def epsilon(n):
    """
    Returns the totally antisymmetric tensor epsilon for n dimensions.
    """
    e = np.zeros([n] * n)
    for idx in np.ndindex(e.shape):
        if len(set(idx)) == n:
            perm = np.array(idx)
            sign = 1
            for i in range(n):
                for j in range(i + 1, n):
                    if perm[i] > perm[j]:
                        perm[i], perm[j] = perm[j], perm[i]
                        sign *= -1
            e[idx] = sign
    return e

