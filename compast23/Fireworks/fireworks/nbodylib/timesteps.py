"""
====================================================================================================================
Collection of functions to estimate the timestep of the Nbody integrations (:mod:`fireworks.nbodylib.timesteps`)
====================================================================================================================

This module contains functions and utilities to estimate the timestep for the Nbody integrations.
There are no strict requirements for these functions. Obviously  it is important that they return a timestep.
It could be also useful to have as inputs a minimum and maximum timestep


"""
from typing import Optional, Tuple, Callable, Union
import numpy as np
import numpy.typing as npt
from ..particles import Particles

def adaptive_timestep_simple(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the position and the velocity of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the R/V of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles

    ts = np.nanmin(particles.radius()/particles.vel_mod())

    # Check tmin, tmax
    if tmin is not None: ts=max(ts,tmin)
    if tmax is not None: ts=min(ts,tmax)

    return ts

def adaptive_timestep_vel_acc(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the velocity and the acceleration of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the V/A of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles
    eta=0.005

    if particles.acc is None:
        ts=0.01
    else:
        ts = eta*np.nanmin(particles.vel_mod()/np.linalg.norm(particles.acc,axis=1))

    # Check tmin, tmax
    if tmin is not None: ts=max(ts,tmin)
    if tmax is not None: ts=min(ts,tmax)

    return ts

def adaptive_timestep_pos_acc(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Very simple adaptive timestep based on the ratio between the position and the acceleration of the particles

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    # Simple idea, use the sqrt(R/A) of the particles to have an estimate of the required timestep
    # Take the minimum among all the particles
    eta = 0.005

    if particles.acc is None:
        ts=0.01
    else:
        ts = eta*np.nanmin(particles.radius()/np.linalg.norm(particles.acc,axis=1))

    # Check tmin, tmax
    if tmin is not None: ts=max(ts,tmin)
    if tmax is not None: ts=min(ts,tmax)

    return ts

def adaptive_timestep_symm(particles: Particles, tmin: Optional[float] = None, tmax: Optional[float] = None) -> float:
    """
    Time step calculated h_sym = 1/2 [h(t) + h(t + h(t))] where h(t+h(t)) is calculated via the derivative of h(t)

    :param particles: that is an instance of the class :class:`~fireworks.particles.Particles`
    :return: estimated timestep
    """

    ##to be implemented

    return t_sym