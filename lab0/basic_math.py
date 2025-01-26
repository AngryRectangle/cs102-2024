import numpy as np
import scipy as sc


def matrix_multiplication(a, b):
    if len(a[0]) != len(b):
        raise ValueError
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
    common = []
    for coeffs in [c1, c2]:
        a, b, c = coeffs
        discriminant = b**2 - 4 * a * c
        if discriminant == 0:
            root = -b / (2 * a)
            common.append((root, a * root**2 + b * root + c))
        elif discriminant > 0:
            root1 = (-b + discriminant**0.5) / (2 * a)
            root2 = (-b - discriminant**0.5) / (2 * a)
            common.append((root1, a * root1**2 + b * root1 + c))
            common.append((root2, a * root2**2 + b * root2 + c))
    return common

def skew(x):
    n = len(x)
    mean_x = sum(x) / n
    m3 = 0
    for xi in x:
        m3 += (xi - mean_x)**3
    m3 /= n
    m2 = 0
    for xi in x:
        m2 += (xi - mean_x)**2
    m2 /= n
    sigma = m2**0.5
    return round(m3 / sigma**3, 2)

def kurtosis(x):
    n = len(x)
    mean_x = sum(x) / n
    m4 = 0
    for xi in x:
        m4 += (xi - mean_x)**4
    m4 /= n
    m2 = 0
    for xi in x:
        m2 += (xi - mean_x)**2
    m2 /= n
    sigma = m2**0.5
    return round(m4 / sigma**4 - 3, 2)
