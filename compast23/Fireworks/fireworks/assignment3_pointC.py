import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fireworks.ic as fic
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.timesteps as ftim

#Simulation for trajectory
def simulate_trajectory(integrator, planets, dt, dynamics):
    position = []
    velocity = []
    time = []
    acceleration = []
    potential = []
    total_energy = []
    error = []
    t = 0.

    velocity.append(planets.vel.copy())
    position.append(planets.pos.copy())
    initial_acc = dynamics(planets)
    acceleration.append(initial_acc[0].copy() if initial_acc is not None else np.zeros_like(planets.pos[0]))
    total_energy.append(planets.Etot()[0])
    error.append(np.zeros_like(planets.Etot()[0]))
    time.append(t)

    Tperiod = np.sqrt(np.sum(planets.mass) / np.linalg.norm(planets.pos[0] - planets.pos[1]) ** 3)
    

    while t < 1000*Tperiod:
        t += dt
        planets, _, acc, _, pot = integrator(particles=planets, tstep=dt, acceleration_estimator=dynamics)
        Etot = planets.Etot()[0]
        Error = (Etot - total_energy[-1])/total_energy[-1]
        position.append(planets.pos.copy())
        velocity.append(planets.vel.copy())
        acceleration.append(acc.copy() if acc is not None else np.zeros_like(planets.pos[0]))
        potential.append(pot)
        total_energy.append(Etot)
        error.append(Error)

        time.append(t)

    position = np.array(position)
    velocity = np.array(velocity)
    acceleration = np.array(acceleration, dtype=object)
    potential = np.array(potential)
    time = np.array(time)
    total_energy = np.array(total_energy)
    error = np.array(error)

    return position, velocity, acceleration, potential, time, total_energy, error

def simulate_trajectory_tsunami(integrator, planets, adaptive_dt):
    position = []
    velocity = []
    time = []
    acceleration = []
    potential = []
    total_energy = []
    error = []
    t = 0.

    velocity.append(planets.vel.copy())
    position.append(planets.pos.copy())
    initial_acc = None
    acceleration.append(initial_acc[0].copy() if initial_acc is not None else np.zeros_like(planets.pos[0]))
    total_energy.append(planets.Etot()[0])
    error.append(np.zeros_like(planets.Etot()[0]))
    time.append(t)

    Tperiod = np.sqrt(np.sum(planets.mass) / np.linalg.norm(planets.pos[0] - planets.pos[1]) ** 3)

    while t < 1000*Tperiod:
        planets, efftime,_,_,pot=integrator(planets,dt)
        t += efftime
        #print("Value of dt evolved:\t" + str(efftime))
        
        Etot = planets.Etot()[0]
        Error = (Etot - total_energy[-1])/total_energy[-1]
        position.append(planets.pos.copy())
        velocity.append(planets.vel.copy())
        acceleration.append(planets.acc.copy() if planets.acc is not None else np.zeros_like(planets.pos[0]))
        potential.append(pot)
        total_energy.append(Etot)
        error.append(Error)

        time.append(t)

    position = np.array(position)
    velocity = np.array(velocity)
    acceleration = np.array(acceleration, dtype=object)
    potential = np.array(potential)
    time = np.array(time)
    total_energy = np.array(total_energy)
    error = np.array(error)

    return position, velocity, acceleration, potential, time, total_energy, error

# Defining plot
def plot_trajectory(position, ax, label):
    x_particle1, y_particle1, z_particle1 = position[:, 0, 0], position[:, 0, 1], position[:, 0, 2]
    x_particle2, y_particle2, z_particle2 = position[:, 1, 0], position[:, 1, 1], position[:, 1, 2]

    ax.plot(x_particle1, y_particle1, z_particle1, marker=',', label=f'Particle 1 - {label}')
    ax.plot(x_particle2, y_particle2, z_particle2, marker=',', label=f'Particle 2 - {label}')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.legend(fontsize='small')

def plot_energy(total_energy, ax, label):
    ax.plot(t, total_energy, marker=',', label=f'Particle 1 - {label}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')

    ax.legend(fontsize='small')

def plot_energy_loss(error, ax, label):
    ax.plot(t, error, marker=',', label=f'Particle 1 - {label}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy Error')

    ax.legend(fontsize='small')

############################# Main #####################################
dt = 0.01
# Different eccentricities
eccentricities = [0.5, 0.9, 0.99]
# Different integrators
integrators = [fint.euler_integrator, fint.velocity_verlet_integrator,
               fint.runge_kutta_4_integrator, fint.symplectic_leapfrog_integrator, fint.integrator_tsunami]
# Creating subplots
for e in eccentricities:
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    fig_tot_energy, axes_tot_energy = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes_tot_energy = axes_tot_energy.flatten()
    
    fig_error,axes_loss = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes_loss = axes_loss.flatten()

    planets = fic.ic_two_body(mass1=1., mass2=1., rp=2., e=e)
    
    for integrator, ax, ax_en,ax_loss in zip(integrators, axes, axes_tot_energy,axes_loss):
        if(integrator.__name__=="integrator_tsunami"):
            position, _, _, _, t, tot_en,error = simulate_trajectory_tsunami(fint.integrator_tsunami, planets.copy(), dt)
            plot_trajectory(position, ax, f'{integrator.__name__}')
            plot_energy(tot_en,ax_en, f'{integrator.__name__}')
            plot_energy_loss(error,ax_loss,f'{integrator.__name__}')
        else:
            position, _, acc, _, t, tot_en,error = simulate_trajectory(integrator, planets.copy(), dt, fdyn.acceleration_pyfalcon)
            plot_trajectory(position, ax, f'{integrator.__name__}')
            plot_energy(tot_en,ax_en, f'{integrator.__name__}')
            plot_energy_loss(error,ax_loss,f'{integrator.__name__}')

    fig.suptitle(f'Trajectories for e={e}', fontsize=16)
    plt.tight_layout()
    fig.savefig(f"Trajectories_e_{e}.png")
    fig_tot_energy.savefig(f"TotalEnergy_e_{e}.png")
    fig_error.savefig(f"Energy_loss_e_{e}.png")
    