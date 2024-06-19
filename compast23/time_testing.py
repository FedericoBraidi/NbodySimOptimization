import time
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
import fireworks.ic as fic
from fireworks.particles import Particles

min_N = 1
max_N = 10
N = np.linspace(min_N,max_N,max_N-min_N+1)
functions = [fdyn.acceleration_pyfalcon,fdyn.acceleration_direct_vectorized,fdyn.acceleration_direct]
ic = []

for n in N:
    ic.append(fic.ic_random_uniform(int(n),min_pos=10.,max_pos=100.,min_vel=10.,max_vel=100.,min_mass=10.,max_mass=100.))

for func in functions:
    for i in range(len(N)):
        t1 = time.perf_counter()
        func(ic[i])
        t2 = time.perf_counter()
        print("Execution time for " + str(int(N[i])) + " particles with fuction " + func.__name__ + ": " + str(t2-t1))
