import numpy as np

# Quasi-Newton Method to find the minimum
def qnm(f, s, h=0.0001):
    while True:
        gradient = [(f(s[0]+h, s[1]) - f(s[0], s[1])) / h, (f(s[0], s[1]+h) - f(s[0], s[1])) / h]

        hessian = [[(f(s[0]+2*h, s[1]) - 2*f(s[0]+h, s[1]) + f(s[0], s[1])) / h**2,
                    (f(s[0] + h, s[1] + h) - f(s[0] + h, s[1]) - f(s[0], s[1] + h) + f(s[0], s[1])) / h ** 2],
                   [(f(s[0]+h, s[1]+h) - f(s[0]+h, s[1]) - f(s[0], s[1]+h) + f(s[0], s[1])) / h**2,
                    (f(s[0], s[1]+2*h) - 2*f(s[0], s[1]+h) + f(s[0], s[1])) / h**2]]

        d = np.dot(np.linalg.inv(hessian), np.array(gradient))
        if np.linalg.norm(d) < h: break
        s = np.array(s) - d
    return tuple(s)
