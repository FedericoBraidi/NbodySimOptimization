"""
==============================================================
Initial conditions utilities , (:mod:`fireworks.ic`)
==============================================================

This module contains functions and utilities to generate
initial conditions for the Nbody simulations.
The basic idea is that each function/class should returns
an instance of the class :class:`~fireworks.particles.Particles`

"""

import numpy as np
import tensorflow as tf
from .particles import Particles,Particles_tf
# from numba import njit


tf.config.optimizer.set_jit(True)
__all__ = ['ic','ic_tf']

class ic:
    
    def ic_random_uniform(N: int, min_pos: float, max_pos: float, min_vel: float, max_vel: float, min_mass: float, max_mass:
                          float, seed: int=None) -> Particles:
        """
        Generate random initial condition drawing from a uniform distribution
        for the position, velocity and mass.
    
        :param N: number of particles to generate
        :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
        """
        if seed is not None:
            np.random.seed(seed)
        pos  = np.random.uniform(low=min_pos,high=max_pos,size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
        vel  = np.random.uniform(low=min_vel,high=max_vel,size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
        mass  = np.random.uniform(low=min_mass,high=max_mass,size=1*N) # Generate Nx1 1D array
        
        return Particles(position=pos, velocity=vel, mass=mass)
   
    def ic_random_normal(N: int, mass: float=1, seed: int=None) -> Particles:
        """
        Generate random initial condition drawing from a normal distribution
        (centred in 0 and with dispersion 1) for the position and velocity.
        The mass is instead the same for all the particles.
    
        :param N: number of particles to generate
        :param mass: mass of the particles
        :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
        """
        if seed is not None:
            np.random.seed(seed)
        pos  = np.random.normal(size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
        vel  = np.random.normal(size=3*N).reshape(N,3) # Generate 3xN 1D array and then reshape as a Nx3 array
        mass = np.ones(N)*mass
    
        return Particles(position=pos, velocity=vel, mass=mass)
    
   
    def ic_two_body(mass1: float, mass2: float, rp: float, e: float):
        """
        Create initial conditions for a two-body system.
        By default the two bodies will placed along the x-axis at the
        closest distance rp.
        Depending on the input eccentricity the two bodies can be in a
        circular (e<1), parabolic (e=1) or hyperbolic orbit (e>1).
    
        :param mass1:  mass of the first body [nbody units]
        :param mass2:  mass of the second body [nbody units]
        :param rp: closest orbital distance [nbody units]
        :param e: eccentricity
        :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
        """
    
        Mtot=mass1+mass2
    
        if e==1.:
            vrel=np.sqrt(2*Mtot/rp)
        else:
            a=rp/(1-e)
            vrel=np.sqrt(Mtot*(2./rp-1./a))
    
        # To take the component velocities
        # V1 = Vcom - m2/M Vrel
        # V2 = Vcom + m1/M Vrel
        # we assume Vcom=0.
        v1 = -mass2/Mtot * vrel
        v2 = mass1/Mtot * vrel
    
        pos  = np.array([[0.,0.,0.],[rp,0.,0.]])
        vel  = np.array([[0.,v1,0.],[0.,v2,0.]])
        mass = np.array([mass1,mass2])
    
        return Particles(position=pos, velocity=vel, mass=mass)

class ic_tf:
    
    def ic_random_uniform(N: int, min_pos: float, max_pos: float, min_vel: float, max_vel: float, min_mass: float, max_mass: float, seed:
                          int=None) -> Particles_tf:
        """
        Generate random initial conditions drawing from a uniform distribution for the position, velocity, and mass.

        N: Number of particles to generate.
        min_pos: Minimum value for position.
        max_pos: Maximum value for position.
        min_vel: Minimum value for velocity.
        max_vel: Maximum value for velocity.
        min_mass: Minimum value for mass.
        max_mass: Maximum value for mass.
        seed: Seed for random number generation.
        return: An instance of the class `Particles_tf` containing the generated particles.
        """
        if seed is not None:
            tf.random.set_seed(seed)
        pos = tf.random.uniform(shape=(N, 3), minval=min_pos, maxval=max_pos, dtype=tf.float32)
        vel = tf.random.uniform(shape=(N, 3), minval=min_vel, maxval=max_vel, dtype=tf.float32)
        mass = tf.random.uniform(shape=(N,), minval=min_mass, maxval=max_mass, dtype=tf.float32)
        return Particles_tf(position=pos, velocity=vel, mass=mass)
    
    def ic_random_normal(N: int, mass: float=1, seed: int=None) -> Particles_tf:
        """
        Generate random initial conditions drawing from a normal distribution for the position and velocity.
        The mass is the same for all particles.

        :param N: Number of particles to generate.
        :param mass: Mass of the particles.
        :param seed: Seed for random number generation.
        :return: An instance of the class `Particles_tf` containing the generated particles.
        """
        if seed is not None:
            tf.random.set_seed(seed)
        pos = tf.random.normal(shape=(N, 3), dtype=tf.float32)
        vel = tf.random.normal(shape=(N, 3), dtype=tf.float32)
        mass = tf.ones(shape=(N,), dtype=tf.float32) * mass
        return Particles_tf(position=pos, velocity=vel, mass=mass)
    
    def ic_two_body(mass1: float, mass2: float, rp: float, e: float) -> Particles_tf:
        """
        Create initial conditions for a two-body system.
        By default, the two bodies will be placed along the x-axis at the closest distance rp.
        Depending on the input eccentricity, the two bodies can be in a circular (e<1), parabolic (e=1), or hyperbolic orbit (e>1).
    
        :param mass1: Mass of the first body.
        :param mass2: Mass of the second body.
        :param rp: Closest orbital distance.
        :param e: Eccentricity.
        :return: An instance of the class `Particles_tf` containing the generated particles.
        """
        if not isinstance(e, float):
            raise ValueError("Eccentricity (e) must be a float.")
    
        Mtot = mass1 + mass2
        if e == 1.:
            vrel = tf.sqrt(2 * Mtot / rp)
        else:
            a = rp / (1 - e)
            vrel = tf.sqrt(Mtot * (2. / rp - 1. / a))
        v1 = -mass2 / Mtot * vrel
        v2 = mass1 / Mtot * vrel
        v1_np = v1.numpy()
        v2_np = v2.numpy()
        pos = tf.constant([[0., 0., 0.], [rp, 0., 0.]], dtype=tf.float32)
        vel = tf.constant([[0., v1_np, 0.], [0., v2_np, 0.]], dtype=tf.float32)
        mass = tf.constant([mass1, mass2], dtype=tf.float32)
        return Particles_tf(position=pos, velocity=vel, mass=mass)




