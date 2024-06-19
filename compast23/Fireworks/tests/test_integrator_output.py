import pytest
import numpy as np
import fireworks.nbodylib.dynamics as fdyn
from fireworks.particles import Particles
from fireworks.ic import ic_random_normal
import fireworks.nbodylib.integrators as fint



def simple_test(integrator):
    """
    Simple test template to test an integrator
    The idea is very simple, assume an acceleration function that returns always 0
    therefore the after an integration step the velocity remains the same
    and the position is pos+vel*dt where dt is the effective integration timestep

    """

    N = 2
    par = ic_random_normal(N,mass=1, seed=42)
    tstep = 1

    par_old = par.copy() # Copy the initial paramter
    print(par.vel[0])
    print(par.pos[0])
    acc_fake = lambda particles, softening: (np.zeros_like(particles.pos), None, None)


    par,teffective,_,_,_=integrator(particles=par,
                                               tstep=tstep,
                                               acceleration_estimator=acc_fake,
                                               softening=0.)
    pos_test = par_old.pos + par_old.vel*teffective

    return(par.vel[0],pos_test[0])
print("Leap frog:\n",simple_test(fint.symplectic_leapfrog_integrator))
print("Euler:\n",simple_test(fint.euler_integrator))    
print("Velocity_verlet:\n",simple_test(fint.velocity_verlet_integrator))
print("Runge_kutta_4:\n",simple_test(fint.runge_kutta_4_integrator))



