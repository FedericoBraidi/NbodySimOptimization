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
tf.config.optimizer.set_jit(True)
__all__ = ['Particles','Particles_tf']
# from numba import njit
from numba import njit, prange



# class Particles:
#     """
#     Simple class to store the properties position, velocity, mass of the particles.
#     Example:

#     >>> from fireworks.particles import Particles
#     >>> position=np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
#     >>> velocity=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
#     >>> mass=np.array([1.,1.,1.])
#     >>> P=Particles(position,velocity,mass)
#     >>> P.pos # particles'positions
#     >>> P.vel # particles'velocities
#     >>> P.mass # particles'masses
#     >>> P.ID # particles'unique IDs

#     The class contains also methods to estimate the radius of all the particles (:func:`~Particles.radius`),
#     the module of the velociy of all the particles (:func:`~Particles.vel_mod`), and the module the positition and
#     velocity of the centre of mass (:func:`~Particles.com_pos` and :func:`~Particles.com_vel`)

#     >>> P.radius() # return a Nx1 array with the particle's radius
#     >>> P.vel_mod() # return a Nx1 array with the module of the particle's velocity
#     >>> P.com() # array with the centre of mass position (xcom,ycom,zcom)
#     >>> P.com() # array with the centre of mass velocity (vxcom,vycom,vzcom)

#     It is also possibile to set an acceleration for each particle, using the method set_acc
#     Example:

#     >>> acc= some_method_to_estimate_acc(P.position)
#     >>> P.set_acc(acc)
#     >>> P.acc # particles's accelerations

#     Notice, if never initialised, P.acc is equal to None

#     The class can be used also to estimate the total, kinetic and potential energy of the particles
#     using the methods :func:`~Particles.Etot`, :func:`~Particles.Ekin`, :func:`~Particles.Epot`
#     **NOTICE:** these methods need to be implemented by you!!!

#     The method :func:`~Particles.copy` can be used to be obtaining a safe copy of the current
#     Particles instances. Safe means that changing the members of the copied version will not
#     affect the members or the original instance
#     Example

#     >>> P=Particles(position,velocity,mass)
#     >>> P2=P.copy()
#     >>> P2.pos[0] = np.array([10,10,10]) # P.pos[0] will not be modified!

#     """
#     def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], mass: npt.NDArray[np.float64]):
#         """
#         Class initialiser.
#         It assigns the values to the class member pos, vel, mass and ID.
#         ID is just a sequential integer number associated to each particle.

#         :param position: A Nx3 numpy array containing the positions of the N particles
#         :param velocity: A Nx3 numpy array containing the velocity of the N particles
#         :param mass: A Nx1 numpy array containing the mass of the N particles
#         """

#         self.pos = np.array(np.atleast_2d(position), dtype=float)
#         if self.pos.shape[1] != 3: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

#         self.vel = np.array(np.atleast_2d(velocity), dtype=float)
#         if self.vel.shape[1] != 3: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
#         if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elemnts")

#         self.mass = np.array(np.atleast_1d(mass), dtype=float)
#         if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elemnts")

#         self.ID=np.arange(len(self.mass), dtype=int)

#         self.acc=None

#     def set_acc(self, acceleration: npt.NDArray[np.float64]):
#         """
#         Set the particle's acceleration

#         :param acceleration: A Nx3 numpy array containing the acceleration of the N particles
#         """

#         acc = np.atleast_2d(acceleration)
#         if acceleration.shape[1] != 3: print(f"Input acceleration should contain a Nx3 array, current shape is {acc.shape}")

#         self.acc=acc

#     def radius(self) -> npt.NDArray[np.float64]:
#         """
#         Estimate the particles distance from the origin of the frame of reference.

#         :return:  a Nx1 array containing the particles' distance from the origin of the frame of reference.
#         """

#         return np.sqrt(np.sum(self.pos*self.pos, axis=1))[:,np.newaxis]

#     def vel_mod(self) -> npt.NDArray[np.float64]:
#         """
#         Estimate the module of the velocity of the particles

#         :return: a Nx1 array containing the module of the particles's velocity
#         """

#         return np.sqrt(np.sum(self.vel*self.vel, axis=1))[:,np.newaxis]

#     def acc_mod(self) -> npt.NDArray[np.float64]:
#         """
#         Estimate the module of the acceleration of the particles

#         :return: a Nx1 array containing the module of the particles's acceleration
#         """

#         return np.sqrt(np.sum(self.acc*self.acc, axis=1))[:,np.newaxis]

#     def com_pos(self) -> npt.NDArray[np.float64]:
#         """
#         Estimate the position of the centre of mass

#         :return: a numpy  array with three elements corresponding to the centre of mass position
#         """

#         return np.sum(self.mass*self.pos.T,axis=1)/np.sum(self.mass)

#     def com_vel(self) -> npt.NDArray[np.float64]:
#         """
#         Estimate the velocity of the centre of mass

#         :return: a numpy  array with three elements corresponding to centre of mass velocity
#         """

#         return np.sum(self.mass*self.vel.T,axis=1)/np.sum(self.mass)

#     def Ekin(self) -> float:
#         """
#         Estimate the total potential energy of the particles:
#         Ekin=0.5 sum_i mi vi*vi

#         :return: total kinetic energy
#         """
#         vel = self.vel
#         mass = self.mass
#         E_kin = np.sum(0.5*mass*np.sum(vel*vel, axis=1))
#         return E_kin

#     def Epot(self,softening: float = 0.,G: float = 1.) -> float:
#         """
#         Estimate the total potential energy of the particles:
#         Epot=-0.5 sumi sumj mi*mj / sqrt(rij^2 + eps^2)
#         where eps is the softening parameter

#         :param softening: Softening parameter 
#         :G is the gravitational constant. Default is 1.
#         :return: The total potential energy of the particles
#         """
#         mass = self.mass
#         pos = self.pos
#         N = len(mass)
#         E_pot = 0.
#         for i in range(N):
#             for j in range(i + 1, N):
#             # Distances between particles i and j
#                 dr = pos[i] - pos[j]
#                 r_ij = np.linalg.norm(dr)
#             # Mass
#                 m_i,m_j = mass[i],mass[j]
#             # Potential Energy
#                 PE = -(m_i*m_j)/np.sqrt(r_ij**2 + softening**2)
#                 E_pot += PE

#         return E_pot

#     def Etot(self,softening: float = 0.) -> tuple[float,float,float]:
#         """
#         Estimate the total  energy of the particles: Etot=Ekintot + Epottot

#         :param softening: Softening parameter
#         :return: a tuple with

#             - Total energy
#             - Total kinetic energy
#             - Total potential energy
#         """

#         Ekin = self.Ekin()
#         Epot = self.Epot(softening=softening)
#         Etot = Ekin + Epot

#         return Etot, Ekin, Epot

#     def copy(self) -> Particles:
#         """
#         Return a copy of this Particle class

#         :return: a copy of the Particle class
#         """

#         par=Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))
#         if self.acc is not None: par.acc=np.copy(self.acc)

#         return Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))

#     def __len__(self) -> int:
#         """
#         Special method to be called when  this class is used as argument
#         of the Python built-in function len()
#         :return: Return the number of particles
#         """

#         return len(self.mass)

#     def __str__(self) -> str:
#         """
#         Special method to be called when  this class is used as argument
#         of the Python built-in function print()
#         :return: short info message
#         """

#         return f"Instance of the class Particles\nNumber of particles: {self.__len__()}"

#     def __repr__(self) -> str:

#         return self.__str__()

class Particles:
    
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
    
        # Exclude self-interaction terms and sum the result
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

# class Particles_Numba:
#     @njit
#     def __init__(self):
#         self.pos = types.float64[:, :]
#         self.vel = types.float64[:, :]
#         self.mass = types.float64[:]
#         self.ID = types.int64[:]
#         self.acc = types.optional(types.float64[:, :])

#     def set_acc(self, acceleration: np.ndarray) -> None:
#         self.acc = np.array(acceleration, dtype=np.float64)

#     def radius(self) -> np.ndarray:
#         return np.sqrt(np.sum(self.pos * self.pos, axis=1))

#     def vel_mod(self) -> np.ndarray:
#         return np.sqrt(np.sum(self.vel * self.vel, axis=1))

#     def com_pos(self) -> np.ndarray:
#         return np.sum(self.mass * self.pos.T, axis=1) / np.sum(self.mass)

#     def com_vel(self) -> np.ndarray:
#         return np.sum(self.mass * self.vel.T, axis=1) / np.sum(self.mass)

#     @staticmethod
#     @njit(nopython=True)
#     def Epot_numba(pos, mass, softening):
#         N = len(mass)
#         E_pot = 0.
#         for i in prange(N):  # Use prange for parallelization
#             for j in range(i + 1, N):  # Keep inner loop as range
#                 # Distances between particles i and j
#                 dr = pos[i] - pos[j]
#                 r_ij = np.linalg.norm(dr)
#                 # Mass
#                 m_i, m_j = mass[i], mass[j]
#                 # Potential Energy
#                 PE = -(m_i * m_j) / np.sqrt(r_ij**2 + softening**2)
#                 E_pot += PE
#         return E_pot
    
#     def Ekin(self) -> float:
#         return 0.5 * np.sum(self.mass * np.sum(self.vel * self.vel, axis=1))

#     def Epot(self, softening: float = 0., G: float = 1.) -> float:
#         return self.Epot_numba(self.pos, self.mass, softening)

#     def Etot(self, softening: float = 0.) -> Tuple[float, float, float]:
#         Ekin = self.Ekin()
#         Epot = self.Epot(softening=softening)
#         Etot = Ekin + Epot
#         return Etot, Ekin, Epot

#     def copy(self) -> 'ParticlesNumba':
#         return ParticlesNumba(self.pos.copy(), self.vel.copy(), self.mass.copy())

#     def __len__(self) -> int:
#         return len(self.mass)

#     def __str__(self) -> str:
#         return f"Instance of the class ParticlesNumba\nNumber of particles: {len(self)}"

#     def __repr__(self) -> str:
#         return self.__str__()