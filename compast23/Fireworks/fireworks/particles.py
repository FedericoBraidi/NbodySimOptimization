"""
==============================================================
Particles data structure , (:mod:`fireworks.particles`)
==============================================================

This module contains the class used to store the Nbody particles data


"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import tensorflow as tf
# tf.config.optimizer.set_jit(True)
__all__ = ['Particles','Particles_tf']
# from numba import njit
# from numba import njit, prange

class Particles:
    
    """
    Simple class to store the properties position, velocity, mass of the particles.
    Example:

    >>> from fireworks.particles import Particles
    >>> position=np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
    >>> velocity=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    >>> mass=np.array([1.,1.,1.])
    >>> P=Particles(position,velocity,mass)
    >>> P.pos # particles'positions
    >>> P.vel # particles'velocities
    >>> P.mass # particles'masses
    >>> P.ID # particles'unique IDs

    The class contains also methods to estimate the radius of all the particles (:func:`~Particles.radius`),
    the module of the velociy of all the particles (:func:`~Particles.vel_mod`), and the module the positition and
    velocity of the centre of mass (:func:`~Particles.com_pos` and :func:`~Particles.com_vel`)

    >>> P.radius() # return a Nx1 array with the particle's radius
    >>> P.vel_mod() # return a Nx1 array with the module of the particle's velocity
    >>> P.com() # array with the centre of mass position (xcom,ycom,zcom)
    >>> P.com() # array with the centre of mass velocity (vxcom,vycom,vzcom)

    It is also possibile to set an acceleration for each particle, using the method set_acc
    Example:

    >>> acc= some_method_to_estimate_acc(P.position)
    >>> P.set_acc(acc)
    >>> P.acc # particles's accelerations

    Notice, if never initialised, P.acc is equal to None

    The class can be used also to estimate the total, kinetic and potential energy of the particles
    using the methods :func:`~Particles.Etot`, :func:`~Particles.Ekin`, :func:`~Particles.Epot`
    **NOTICE:** these methods need to be implemented by you!!!

    The method :func:`~Particles.copy` can be used to be obtaining a safe copy of the current
    Particles instances. Safe means that changing the members of the copied version will not
    affect the members or the original instance
    Example

    >>> P=Particles(position,velocity,mass)
    >>> P2=P.copy()
    >>> P2.pos[0] = np.array([10,10,10]) # P.pos[0] will not be modified!

    """
    def __init__(self, position: np.ndarray, velocity: np.ndarray, mass: np.ndarray):
        self.pos = np.array(position, dtype=np.float32)
        self.vel = np.array(velocity, dtype=np.float32)
        self.mass = np.array(mass, dtype=np.float32)
        self.ID = np.arange(len(mass), dtype=np.int32)
        self.acc = None

    def set_acc(self, acceleration: np.ndarray) -> None:
        self.acc = np.array(acceleration, dtype=np.float32)

    def radius(self) -> np.ndarray[np.float32]:
        return np.sqrt(np.sum(self.pos * self.pos, axis=1))

    def vel_mod(self) -> np.ndarray[np.float32]:
        return np.sqrt(np.sum(self.vel * self.vel, axis=1))

    def com_pos(self) -> np.ndarray[np.float32]:
        return np.sum(self.mass * self.pos.T, axis=1) / np.sum(self.mass)

    def com_vel(self) -> np.ndarray:
        return np.sum(self.mass * self.vel.T, axis=1) / np.sum(self.mass)
    
   
    def Epot(self, softening: float = 0., G: float = 1.) -> float:
        mass = self.mass
        pos = self.pos
        N = len(mass)
        E_pot = 0.
        for i in range(N):
            for j in range(i + 1, N):
                # Distances between particles i and j
                dr = pos[i] - pos[j]
                r_ij = np.linalg.norm(dr)
                # Mass
                m_i, m_j = mass[i], mass[j]
                # Potential Energy
                PE = -(m_i * m_j) / np.sqrt(r_ij**2 + softening**2)
                E_pot += PE
        return E_pot
    
    def Epot_vec(self, softening: float = 0., G: float = 1.) -> float:
        mass = self.mass
        pos = self.pos
        N = len(mass)
        E_pot = 0.
    
        # Create a matrix of relative positions
        dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # Calculate the distances between all pairs of particles
        r_ij = np.linalg.norm(dr, axis=-1)
        
        # Mask the diagonal and upper triangle (i < j)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
        # Calculate the potential energy for all pairs using vectorized operations
        PE = -(mass[:, np.newaxis] * mass[np.newaxis, :]) / np.sqrt(r_ij**2 + softening**2)
        
        # Sum the potential energy contributions
        E_pot = np.sum(PE[mask])
        
        return E_pot


    
    def Ekin(self) -> float:
        return 0.5 * np.sum(self.mass * np.sum(self.vel * self.vel, axis=1))

    # def Etot(self, softening: float = 0.) -> Tuple[float, float, float]:
    #     Ekin = self.Ekin()
    #     Epot = self.Epot(softening=softening)
    #     Etot = Ekin + Epot
    #     return Etot, Ekin, Epot
    
    def Etot(self, softening: float = 0.) -> Tuple[float, float, float]:
        """
        Estimate the total energy of the particles: Etot = Ekin + Epot
    
        :param softening: Softening parameter
        :return: A tuple containing:
            - Total energy
            - Total kinetic energy
            - Total potential energy
        """
        Ekin: float = self.Ekin()
        Epot: float = self.Epot(softening=softening)
        Etot: float = Ekin + Epot

        return Etot, Ekin, Epot

    def copy(self) -> 'Particles':
        return Particles(self.pos.copy(), self.vel.copy(), self.mass.copy())

    def __len__(self) -> int:
        return len(self.mass)

    def __str__(self) -> str:
        return f"Instance of the class Particles\nNumber of particles: {len(self)}"

    def __repr__(self) -> str:
        return self.__str__()

    

class Particles_tf:
    
    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            print("np---> tf")
            return tf.convert_to_tensor(data, dtype=tf.float32)
        elif isinstance(data, tf.Tensor):
            return tf.cast(data, dtype=tf.float32)
        else:
            raise ValueError("Unsupported data type. Must be either a NumPy array or a TensorFlow tensor.")

    def __init__(self, position, velocity, mass):
        self.pos = self._convert_to_tensor(position)
        self.vel = self._convert_to_tensor(velocity)
        self.mass = self._convert_to_tensor(mass)
        self.ID = tf.range(tf.shape(mass)[0], dtype=tf.int32)
        self.acc = None
    
    def set_acc(self, acceleration):
        self.acc = self._convert_to_tensor(acceleration)

    def radius(self):
        return tf.sqrt(tf.reduce_sum(self.pos * self.pos, axis=1))

    def vel_mod(self):
        return tf.sqrt(tf.reduce_sum(self.vel * self.vel, axis=1))

    def acc_mod(self):
        return tf.sqrt(tf.reduce_sum(self.acc * self.acc, axis=1))

    def com_pos(self):
        return tf.reduce_sum(self.mass * self.pos, axis=0) / tf.reduce_sum(self.mass)

    def com_vel(self):
        return tf.reduce_sum(self.mass * self.vel, axis=0) / tf.reduce_sum(self.mass)

    def Ekin(self):
        return tf.reduce_sum(0.5 * self.mass * tf.reduce_sum(self.vel * self.vel, axis=1))

   
    def Epot(self, softening: float = 0., G: float = 1.):
        mass = self.mass
        pos = self.pos
        N = len(mass)
        E_pot = 0.
        for i in range(N):
            for j in range(i + 1, N):
                # Distances between particles i and j
                dr = pos[i] - pos[j]
                r_ij = tf.norm(dr)
                # Mass
                m_i, m_j = mass[i], mass[j]
                # Potential Energy
                PE = -(m_i * m_j) / tf.sqrt(r_ij**2 + softening**2)
                E_pot += PE
        return E_pot
    
    def Epot_tf(self, softening: float = 0., G: float = 1.):
        pos = self.pos
        mass = self.mass
        N = len(mass)
    
        # Expand dimensions for broadcasting
        pos_i = tf.expand_dims(pos, axis=0)
        pos_j = tf.expand_dims(pos, axis=1)
        mass_i = tf.expand_dims(mass, axis=0)
        mass_j = tf.expand_dims(mass, axis=1)
    
        # Calculate distances between all pairs of particles
        dr = pos_i - pos_j
        r_ij = tf.norm(dr, axis=-1)  # Compute Euclidean norm along the last axis
    
        # Compute potential energy using broadcasting
        PE = -(mass_i * mass_j) / tf.sqrt(r_ij**2 + softening**2)
    
        # Excludes self-interaction terms and sum the result
        mask = tf.math.logical_not(tf.eye(N, dtype=tf.bool))
        E_pot = tf.reduce_sum(tf.boolean_mask(PE, mask))/2
    
        return E_pot
        

    def Etot(self, softening: float = 0.):
        # Implementation using TensorFlow operations
        Ekin = self.Ekin()
        Epot = self.Epot(softening=softening)
        Etot = Ekin + Epot
        return Etot, Ekin, Epot

    def copy(self):
        return Particles_tf(self.pos, self.vel, self.mass)

    def __len__(self):
        return tf.shape(self.mass)[0]

    def __str__(self):
        return f"Instance of the class Particles\nNumber of particles: {self.__len__()}"

    def __repr__(self):
        return self.__str__()

