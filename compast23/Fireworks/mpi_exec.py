from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from fireworks.ic import ic_tf as fic_tf 
from fireworks.ic import ic as fic
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.timesteps as ftim
from tqdm.notebook import tqdm
from fireworks.particles import Particles
import pandas as pd
import cProfile



def mpi_initialize_particles(n_particles, min_pos, max_pos, min_vel, max_vel, min_mass, max_mass, seed):
    particles = fic.ic_random_uniform(
        n_particles, min_pos=min_pos, max_pos=max_pos,
        min_vel=min_vel, max_vel=max_vel, min_mass=min_mass, max_mass=max_mass, seed=seed
    )
    return particles

def mpi_measure_compile_time(particles, facc_list):
    compile_times = []
    
    for facc in facc_list:
        t1 = time.perf_counter()
        
        # Call the acceleration function
        acceleration = facc(particles, softening=1e-10)
        t2 = time.perf_counter()
            
        dt = t2 - t1
        compile_times.append(dt)
        
        print(f"Time taken for acceleration using {facc.__name__}: {dt} seconds")
    
    
    return compile_times

def mpi_test_time_ic_random_uniform(N, min_pos, max_pos, min_vel, max_vel, min_mass, max_mass, seed, facc_list):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    local_results = {}
    for i, n_particles in enumerate(N):
        if i % size == rank:  # Distribute tasks among processes
            particles = mpi_initialize_particles(n_particles, min_pos, max_pos, min_vel, max_vel, min_mass, max_mass, seed)
            compile_times = mpi_measure_compile_time_and_memory(particles, facc_list)
            local_results[n_particles] = {
                'particles': particles,
                'compile_times': compile_times,
            }

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        results = {}
        for result in all_results:
            results.update(result)
        return results

# Defining initial conditions for the test
N = np.linspace(10, 9000, 10).astype(int)
min_pos = 10.
max_pos = 100.
min_vel = 10.
max_vel = 100.
min_mass = 10.
max_mass = 100.
seed = 1

facc_list = [fdyn.acceleration_direct, fdyn.acceleration_direct_vectorized, fdyn.acceleration_pyfalcon]

if __name__ == "__main__":
    results = mpi_test_time_and_memory_ic_random_uniform(N, min_pos, max_pos, min_vel, max_vel, min_mass, max_mass, seed, facc_list)
    if results:
        # Save or print the results
        print(results)

