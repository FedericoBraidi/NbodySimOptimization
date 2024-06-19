# test_energy.py
import numpy as np
import pytest
from fireworks.particles import Particles

#pos = np.array([[0., 0., 0.], [1., 0., 0.]])
#vel = np.ones_like(pos)
#mass = np.ones(len(pos))

#particles = Particles(pos, vel, mass)
#E = particles.Etot()
#print(E)
def test_Etot():
    pos = np.array([[0., 0., 0.], [1., 0., 0.]])
    vel = np.ones_like(pos)
    mass = np.ones(len(pos))

    particles = Particles(pos, vel, mass)

    act_Ekin = 3.
    act_Epot = -1.
    act_Etot = act_Ekin + act_Epot

    # Test the Etot method
    softening = 0.
    pred_Etot, pred_Ekin, pred_Epot = particles.Etot(softening=softening)

    # Check the results
    assert pred_Ekin == pytest.approx(act_Ekin, rel=1e-5)
    assert pred_Epot == pytest.approx(act_Epot, rel=1e-5)
    assert pred_Etot == pytest.approx(act_Etot, rel=1e-5)

