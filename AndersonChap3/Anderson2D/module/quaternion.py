import numpy as np
from numba import njit

"""----------------------------------------------------------------------------------------------------------------
In this class, the __init__ method initializes a quaternion object with coefficients a, b, c, and d.
The __add__ and __sub__ methods implement addition and subtraction of quaternions, and the __mul__ 
method implements multiplication of quaternions. The conjugate method returns the conjugate of a
quaternion, and the norm method returns the norm of a quaternion. The normalize method returns
a normalized quaternion (i.e., a unit quaternion). The __str__ method is used to represent the 
quaternion in string format.
----------------------------------------------------------------------------------------------------------------"""


class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __add__(self, other):
        return Quaternion(
            self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d
        )

    def __sub__(self, other):
        return Quaternion(
            self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d
        )

    def __mul__(self, other):
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        return Quaternion(a, b, c, d)

    def __str__(self):
        return f"({self.a} + {self.b}i + {self.c}j + {self.d}k)"

    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def norm(self):
        return np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

    def normalize(self):
        norm = self.norm()
        return Quaternion(self.a / norm, self.b / norm, self.c / norm, self.d / norm)
