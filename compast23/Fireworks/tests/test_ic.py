import pytest
import numpy as np
import fireworks.ic as fic

def test_ic_random_uniform():
    """
    Simple test for the method ic_random_uniform
    """

    N=100
    min_pos=10.
    max_pos=100.
    min_vel=10.
    max_vel=100.
    min_mass=10.
    max_mass=100.

    particles = fic.ic_random_uniform(N,min_pos=min_pos,max_pos=max_pos,min_vel=min_vel,max_vel=max_vel,min_mass=min_mass,max_mass=max_mass)

    assert len(particles)==N #Test if we create the right amount of particles
    assert (np.min(particles.pos)>=min_pos and np.max(particles.pos<=max_pos)) #Test if the positions are within the boundaries we set
    assert (np.min(particles.vel)>=min_vel and np.max(particles.vel<=max_vel)) #Test if the velocities are within the boundaries we set
    assert (np.min(particles.mass)>=min_mass and np.max(particles.mass<=max_mass)) #Test if the masses are within the boundaries we set

def test_ic_random_normal():
    """
    Simple test for the method ic_random_normal
    """

    N=100
    mass=10.
    particles = fic.ic_random_normal(N,mass=mass)

    assert len(particles)==N #Test if we create the right amount of particles
    assert np.all(particles.mass==10.) #Test if the mass of the particles is set correctly

def test_ic_two_body_circular():
    """
    Simple test for equal mass stars in a circular orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=0.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -1./Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == 1./Mtot

def test_ic_two_body_parabolic():
    """
    Simple test for equal mass stars in a parabolic orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=1.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -np.sqrt(2)/Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == np.sqrt(2)/Mtot

def test_ic_two_body_hyperbolic():
    """
    Simple test for equal mass stars in a hyperbolic orbit
    """
    mass1=1.
    mass2=1.
    Mtot=mass1+mass2
    rp=2.
    particles = fic.ic_two_body(mass1,mass2,rp=rp,e=3.)

    assert pytest.approx(particles.vel[0,1],1e-10) == -2./Mtot
    assert pytest.approx(particles.vel[1,1],1e-10) == 2./Mtot
