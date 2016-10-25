import numpy as np

def var1(n):        #  two-pass algorithm, was slightly faster than .var() method
    avg = n.mean()
    delta = n - avg
    delta_squared_sum = np.dot(delta,delta)
    return delta_squared_sum/n.size

def var2(n):      # this was the fastest but bad for big numbers with small variation. E_n_sq can be very large, avg can be very large. E_n_sq - avg**2 can be very small.
    avg = n.mean()
    n_sq_sum = np.dot(n,n)
    E_n_sq = n_sq_sum/n.size
    return E_n_sq - avg**2

def rs1(n):  # single pass, running statatistics version (Welford algorithm), was much slower but uses less memory, will have advantage for big data or stream of data. Wikipedia says "These formulas suffer from numerical instability." Why?
    k = 1
    M = n[0]
    S = 0
    for ni in n:
        M_old = M
        M = M_old + (ni - M_old) / k
        S = S + (ni - M_old) * (ni - M)
        k += 1
    var = S / (k - 1)
    return M, var

def rs2(n):  # single pass, running statatistics version (Welford algorithm)
    M = n[0]
    S = 0
    for k in range(1,n.size):
        M_old = M
        M = M_old + (n[k] - M_old)/k
        S = S + (n[k]-M_old)*(n[k]-M)
        k += 1
    var = S/(k-1)
    return M, var

def rs3(n):  # single pass, running statatistics version (Welford's online algorithm, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
    i = 0
    M = 0.0
    S = 0.0
    for ni in n:
        i += 1
        delta = ni - M
        M += delta/i
        S += delta*(ni - M)
    if i < 2:
        return n[0], 0
    else:
        return M, S/(i-1)

# timeit(samples.var())    # 28 um per loop
# timeit(var1(samples))    # 25.2 um per loop
# timeit(var2(samples))    # 14 um per loop --> this was the fastest but bad for big numbers with small variation
# timeit(rs1(samples))     # 6.89 mm per loop
# timeit(rs2(samples))     # 8.83 ms per loop
# timeit(rs3(samples))     # 6.41 ms per loop   --> best realization of Welford's algorithm (?)

x = np.random.randn(1000000)
y = np.random.randn(1000000)

z1 = sum(x[i] * y[i] for i in np.arange(len(x)))  # elapsed time: 1.2176 seconds
z2 = np.dot(x, y)  # elapsed time: 0.0045 seconds

np.random.seed(42)
offset = 1e+7          # try different offsets (ex. 0, 1e+14). var2() goes off starting from 1e+7
samples = np.random.randint(0,2,10000) + offset
print('benchmark .mean() method')
print(samples.mean()) # 0.4987
print('benchmark .var() method')
print(samples.var()) # 0.499998309997
print('var1(), two-pass algorithm')
print(var1(samples))
print('var2(), alternative, two-pass algorithm, bad for large numbers with a small variance. try adjusting "offset" in the code.')
print(var2(samples))
print('rs1(), single pass, running statatistics version (Welford algorithm)')
print(rs1(samples))
print('rs2(), single pass, running statatistics version (Welford algorithm)')
print(rs2(samples))
print('rs3(), single pass, running statatistics version (Welford algorithm)')
print(rs3(samples))
