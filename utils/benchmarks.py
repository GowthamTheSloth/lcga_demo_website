# utils/benchmarks.py
import numpy as np

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1.0)**2))

def rastrigin(x):
    x = np.asarray(x)
    A = 10.0
    return float(A * x.size + np.sum(x**2 - A * np.cos(2*np.pi*x)))

def ackley(x):
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2*np.pi
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c*x))
    return float(-a * np.exp(-b*np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e)

def griewank(x):
    x = np.asarray(x)
    part1 = np.sum(x**2)/4000.0
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, x.size+1))))
    return float(part1 - part2 + 1.0)

# mapping
FUNCTIONS = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Griewank": griewank,
}
