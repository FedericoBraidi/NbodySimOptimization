"""
=========================================================
ODE integrators  (:mod:`fireworks.nbodylib.integrators`)
=========================================================

This module contains a collection of integrators to integrate one step of the ODE N-body problem
The functions included in this module should follow the input/output structure
of the template method :func:`~integrator_template`.

All the functions need to have the following input parameters:

    - particles, an instance of the  class :class:`~fireworks.particles.Particles`
    - tstep, timestep to be used to advance the Nbody system using the integrator
    - acceleration_estimator, it needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
        following the input/output style of the template function
        (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    - softening, softening values used to estimate the acceleration
    - external_accelerations, this is an optional input, if not None, it has to be a list
        of additional callable to estimate additional acceleration terms (e.g. an external potential or
        some drag term depending on the particles velocity). Notice that if the integrator uses the jerk
        all this additional terms should return the jerk otherwise the jerk estimate is biased.

Then, all the functions need to return the a tuple with 5 elements:

    - particles, an instance of the  class :class:`~fireworks.particles.Particles` containing the
        updates Nbody properties after the integration timestep
    - tstep, the effective timestep evolved in the simulation (for some integrator this can be
        different wrt the input tstep)
    - acc, the total acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        can be set to None
    - jerk, total time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, total  gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.

"""
from typing import Optional, Tuple, Callable, Union, List, Any
import numpy as np
import numpy.typing as npt
from ..particles import Particles,Particles_tf
import tensorflow as tf

try:
    import tsunami
    tsunami_load=True
except:
    tsunami_load=False

def integrator_template(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None):
    """
    This is an example template of the function you have to implement for the N-body integrators.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct,jerkt,potentialt=ext_acc_estimator(particles,softening)
            acc+=acct
            if jerk is not None and jerkt is not None: jerk+=jerkt
            if potential is not None and potentialt is not None: potential+=potentialt

    #Exemple of an Euler estimate
    particles.pos = particles.pos + particles.vel*tstep # Update pos
    particles.vel = particles.vel + acc*tstep # Update vel
    particles.set_acc(acc) #Set acceleration

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)

    return (particles, tstep, acc, jerk, potential)

def euler_integrator(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None):
    """
    This is an example of an Euler integrator.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct,jerkt,potentialt=ext_acc_estimator(particles,softening)
            acc+=acct
            if jerk is not None and jerkt is not None: jerk+=jerkt
            if potential is not None and potentialt is not None: potential+=potentialt

    #Euler estimate(sympletic)
    particles.vel = particles.vel + acc*tstep # Update vel
    particles.pos = particles.pos + particles.vel*tstep # Update pos
    particles.set_acc(acc) #Set acceleration

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)

    return (particles, tstep, acc, jerk, potential)

def symplectic_leapfrog_integrator(particles: Particles,
                                   tstep: float,
                                   acceleration_estimator: Union[Callable, List],
                                   softening: float = 0.,
                                   external_accelerations: Optional[List] = None):
    """
    This is an example of the Sympletic Leapfrog integrator.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """
    
    acc, jerk, potential = acceleration_estimator(particles, softening)

    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    # Symplectic leapfrog integration
    particles.vel = particles.vel + 0.5 * acc * tstep  # Update velocity at half-step
    particles.pos = particles.pos + particles.vel * tstep  # Update position
    acc_new,jerk_new,potential_new = acceleration_estimator(particles, softening)# Update acceleration at new position
    particles.vel = particles.vel + 0.5 * acc_new * tstep  # Update velocity with the new acceleration
    particles.set_acc(acc_new)
   
    # Now return the updated particles, the acceleration, jerk (can be None), and potential (can be None)
    return particles, tstep, acc_new, jerk, potential


def velocity_verlet_integrator(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.,
                        external_accelerations: Optional[List] = None):
    """
    This is an example of the Velocity Verlet integrator.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """

    acc,jerk,potential=acceleration_estimator(particles,softening)

    #Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct,jerkt,potentialt=ext_acc_estimator(particles,softening)
            acc+=acct
            if jerk is not None and jerkt is not None: jerk+=jerkt
            if potential is not None and potentialt is not None: potential+=potentialt

    #verlet  estimate
    particles.pos = particles.pos + particles.vel*tstep + 0.5*acc*(tstep**2) # Update position
    acc_new,jerk_new,potential_new = acceleration_estimator(particles,softening) # Update acceleration
    particles.vel = particles.vel + 0.5*(acc+acc_new)*tstep # Update vel
    particles.set_acc(acc_new) #Set acceleration

    # Now return the updated particles, the acceleration, jerk (can be None) and potential (can be None)

    return (particles, tstep, acc, jerk, potential)

def runge_kutta_4_integrator(particles: Particles,
                             tstep: float,
                             acceleration_estimator: Union[Callable, List],
                             softening: float = 0.,
                             external_accelerations: Optional[List] = None):
    """
    This is an example of the Runge Kutta 4th order  integrator.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can use smaller sub-time step to
    achieve the final result
    :param acceleration_estimator: It needs to be a function from the module (:mod:`fireworks.nbodylib.dynamics`)
    following the input/output style of the template function  (:func:`fireworks.nbodylib.dynamics.acceleration_estimate_template`).
    :param softening: softening parameter for the acceleration estimate, can use 0 as default value
    :param external_accelerations: a list of additional force estimators (e.g. an external potential field) to
    consider to estimate the final acceleration (and if available jerk) on the particles
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation (for some integrator this can be
            different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle, ca be set to None
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None

    """

    # Helper function to compute the derivative at a given state
    def compute_derivative(state, t, acceleration_function):
        """
        Takes three arguments
        state : current state of the particles
        t : time step
        acceleration_function : the accleration function being used from the module (:mod: 'fireworks.nbodylib.dynamics').

        Returns:
        the velocity
        acceleration 
        """
        #pos, vel = state.pos, state.vel
        acc, jerk, potential = acceleration_function(state, softening)
        return state.vel, acc

    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    # RK4 integration
    current_state = particles.copy()
    #print("previous state:\n",current_state.pos[0],current_state.vel[0],current_state.mass[0])
    
    ################### k1v,k1a ###########################

    k1v, k1a = compute_derivative(current_state, 0, acceleration_estimator)
    #print("k1:\n",k1v,k1a)
    new_state1_pos = current_state.pos + 0.5 * tstep * k1v
    new_state1_vel = current_state.vel + 0.5 * tstep * k1a
    new_state1 = Particles(new_state1_pos,new_state1_vel,current_state.mass)
    #print("new_state1:\n",new_state1.pos[0],new_state1.vel[0],new_state1.mass[0])
   
   ################### k2v,k2a ###########################

    k2v, k2a = compute_derivative(new_state1, tstep / 2, acceleration_estimator)
    #print("k2:\n",k2v,k2a)
    new_state2_pos = current_state.pos + 0.5 * tstep * k2v
    new_state2_vel = current_state.vel + 0.5 * tstep * k2a
    new_state2 = Particles(new_state2_pos,new_state2_vel,current_state.mass)
    #print("new_state2:\n",new_state2.pos[0],new_state2.vel[0],new_state2.mass[0])
    

    ################### k3v,k3a ###########################

    k3v, k3a = compute_derivative(new_state2, tstep / 2, acceleration_estimator)
    #print("k3:\n",k3v,k3a)
    new_state3_pos = current_state.pos +  tstep * k3v
    new_state3_vel = current_state.vel +  tstep * k3a
    new_state3 = Particles(new_state3_pos,new_state3_vel,current_state.mass)
    #print("new_state3:\n",new_state3.pos[0],new_state3.vel[0],new_state3.mass[0])

    ################### k4v,k4a ###########################

    k4v, k4a = compute_derivative(new_state3, tstep, acceleration_estimator)



    particles.pos = current_state.pos + (tstep / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    particles.vel = current_state.vel + (tstep / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)

    acc_new,jerk_new,potential_new = acceleration_estimator(particles, softening)  # Update acceleration at new position
    particles.set_acc(acc_new)

    # Now return the updated particles, the acceleration, jerk (can be None), and potential (can be None)
    return particles, tstep,acc_new,jerk_new,potential_new


def integrator_tsunami(particles: Particles,
                       tstep: float,
                       acceleration_estimator: Optional[Callable]= None,
                       softening: float = 0.,
                       external_accelerations: Optional[List] = None):
    """
    Special integrator that is actually a wrapper of the TSUNAMI integrator.
    TSUNAMI is regularised and it has its own way to estimate the acceleration,
    set the timestep and update the system.
    Therefore in this case tstep should not be the timestep of the integration, but rather
    the final time of our simulations, or an intermediate time in which we want to store
    the properties or monitor the sytems.
    Example:

    >>> tstart=0
    >>> tintermediate=[5,10,15]
    >>> tcurrent=0
    >>> for t in tintermediate:
    >>>     tstep=t-tcurrent
    >>>     particles, efftime,_,_,_=integrator_tsunami(particles,tstep)
    >>>     # Here we can save stuff, plot stuff, etc.
    >>>     tcurrent=tcurrent+efftime

    .. note::
        In general the TSUNAMI integrator is much faster than any integrator with can implement
        in this module.
        However, Before to start the proper integration, this function needs to perform some preliminary
        steps to initialise the TSUNAMI integrator. This can add a  overhead to the function call.
        Therefore, do not use this integrator with too small timestep. Acutally, the best timstep is the
        one that bring the system directly to the final time. However, if you want to save intermediate steps
        you can split the integration time windows in N sub-parts, calling N times this function.

    .. warning::
        It is important to notice that given the nature of the integrator (based on chain regularisation)
        the final time won't be exactly the one put in input. Take this in mind when using this  integrator.
        Notice also that the TSUNAMI integrator will rescale your system to the centre of mass frame of reference.



    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: final time of the current integration
    :param acceleration_estimator: Not used
    :param softening: Not used
    :param external_accelerations: Not used
    :return: A tuple with 5 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation, it wont'be exaxtly the one in input
        - acc, it is None
        - jerk, it is None
        - pot, it is None

    """
    if not tsunami_load: return ImportError("Tsunami is not available")

    code = tsunami.Tsunami(1.0, 1.0)
    code.Conf.dcoll = 0.0 #Disable collisions
    # Disable extra forces (already disabled by default)
    code.Conf.wPNs = False  # PNs
    code.Conf.wEqTides = False  # Equilibrium tides
    code.Conf.wDynTides = False  # Dynamical tides

    r=np.ones_like(particles.mass)
    st=np.array(np.ones(len(particles.mass))*(-1), dtype=int)
    code.add_particle_set(particles.pos, particles.vel, particles.mass, r, st)
    # Synchronize internal code coordinates with the Python interface
    code.sync_internal_state(particles.pos, particles.vel)
    # Evolve system from 0 to tstep - NOTE: the system final time won't be exacly tstep, but close
    code.evolve_system(tstep)
    # Synchronize realt to the real internal system time
    time = code.time
    # Synchronize coordinates to Python interface
    code.sync_internal_state(particles.pos, particles.vel)

    return (particles, time, None, None, None)


def runge_kutta_4_integrator_tf(particles: Particles_tf,
                                tstep: float,
                                acceleration_estimator: Callable,
                                softening: float = 0.,
                                external_accelerations: Optional[List] = None,
                                potential: bool = False) \
        -> Tuple[Particles_tf, float, Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]]:
    """
    This is an example of the Runge Kutta 4th order integrator using TensorFlow operations.
    
    :param particles: Instance of the class Particles_tf
    :param tstep: Time-step for the current integration
    :param acceleration_estimator: Function to estimate acceleration, following the input/output style
                                    of the acceleration function.
    :param softening: Softening parameter for the acceleration estimate, default is 0.
    :param external_accelerations: Additional force estimators to consider for estimating the final acceleration
    :param potential: Whether to compute and return the potential
    :return: A tuple with 5 elements:
                - The updated particles instance
                - The effective timestep evolved in the simulation
                - The final acceleration for each particle (can be None)
                - The time derivative of the acceleration (can be None)
                - The potential at each particle position (can be None)
    """

    def compute_derivative_tf(state, t, acceleration_function):
        """
        Compute the derivative at a given state using TensorFlow operations.
        
        :param state: Current state of the particles
        :param t: Time step
        :param acceleration_function: The acceleration function being used
        :return: The velocity, acceleration, and potential
        """
        acc, _, pot = acceleration_function(state, softening=softening, potential=potential)
        return state.vel, acc, pot

    # Initialize variables
    acc, jerk, pot = acceleration_estimator(particles, softening=softening, potential=potential)
    # print(f"potential inside compute derivative {pot}\n")
    
    # Check additional accelerations
    if external_accelerations is not None:
        for ext_acc_estimator in external_accelerations:
            acct, jerkt, potentialt = ext_acc_estimator(particles, softening)
            acc += acct
            if jerk is not None and jerkt is not None: jerk += jerkt
            if potential is not None and potentialt is not None: potential += potentialt

    # RK4 integration
    current_state = particles.copy()
    # print("previous state:\n",current_state.pos[0],current_state.vel[0],current_state.mass[0])
    
    
            
    k1v, k1a, _ = compute_derivative_tf(current_state, 0, acceleration_estimator)
    new_state1_pos = current_state.pos + 0.5 * tstep * k1v
    new_state1_vel = current_state.vel + 0.5 * tstep * k1a
    new_state1 = Particles_tf(new_state1_pos, new_state1_vel, current_state.mass)

    k2v, k2a, _ = compute_derivative_tf(new_state1, tstep / 2, acceleration_estimator)
    new_state2_pos = current_state.pos + 0.5 * tstep * k2v
    new_state2_vel = current_state.vel + 0.5 * tstep * k2a
    new_state2 = Particles_tf(new_state2_pos, new_state2_vel, current_state.mass)

    k3v, k3a, _ = compute_derivative_tf(new_state2, tstep / 2, acceleration_estimator)
    new_state3_pos = current_state.pos + tstep * k3v
    new_state3_vel = current_state.vel + tstep * k3a
    new_state3 = Particles_tf(new_state3_pos, new_state3_vel, current_state.mass)

    k4v, k4a, _ = compute_derivative_tf(new_state3, tstep, acceleration_estimator)

    # Update particle positions and velocities
    particles.pos = current_state.pos + (tstep / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    particles.vel = current_state.vel + (tstep / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
    # print(f"new state of the particle is pos: {particles.pos} and vel: {particles.vel} \n")
    # print(f"******{particles.Epot_tf(softening =softening)}******")
    # Update acceleration at new position
    acc_new, _, pot_new = acceleration_estimator(particles, softening=softening, potential=potential)
    particles.set_acc(acc_new)
    # print(f"new potential is {pot_new}\n")
    # Return the updated particles, acceleration, jerk, and potential
    return particles, tstep, acc_new, jerk, pot_new