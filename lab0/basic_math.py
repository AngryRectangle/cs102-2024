import numpy as np
import scipy as sc


def matrix_multiplication(a, b):
    if len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions for multiplication")
    
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            s = 0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            row.append(s)
        result.append(row)
    return result


def functions(a, b):
    c1 = list(map(float, a.split()))
    c2 = list(map(float, b.split()))

    if c1 == c2:
        return None

    A = c1[0] - c2[0]
    B = c1[1] - c2[1]
    C = c1[2] - c2[2]
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return []
        else:
            x0 = -C / B
            y0 = c1[0] * x0**2 + c1[1] * x0 + c1[2]
            return [(x0, y0)]

    disc = B**2 - 4*A*C
    if disc < 0:
        return []
    elif abs(disc) < 1e-14:
        x0 = -B / (2*A)
        y0 = c1[0] * x0**2 + c1[1] * x0 + c1[2]
        return [(x0, y0)]
    else:
        sqrt_disc = disc**0.5
        x1 = (-B + sqrt_disc) / (2*A)
        x2 = (-B - sqrt_disc) / (2*A)
        y1 = c1[0] * x1**2 + c1[1] * x1 + c1[2]
        y2 = c1[0] * x2**2 + c1[1] * x2 + c1[2]

        if x1 < x2:
            return [(x1, y1), (x2, y2)]
        else:
            return [(x2, y2), (x1, y1)]


def skew(x):
    n = len(x)
    mean_x = sum(x) / n
    m3 = sum((xi - mean_x)**3 for xi in x) / n
    m2 = sum((xi - mean_x)**2 for xi in x) / n
    sigma = m2**0.5
    return round(m3 / (sigma**3), 2)


def kurtosis(x):
    n = len(x)
    mean_x = sum(x) / n
    m4 = sum((xi - mean_x)**4 for xi in x) / n
    m2 = sum((xi - mean_x)**2 for xi in x) / n
    sigma = m2**0.5
    return round(m4 / (sigma**4) - 3, 2)
