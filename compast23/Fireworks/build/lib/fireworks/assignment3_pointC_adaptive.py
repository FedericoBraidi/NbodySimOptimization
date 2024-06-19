import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fireworks.ic as fic
import fireworks.nbodylib.integrators as fint
import fireworks.nbodylib.dynamics as fdyn
import fireworks.nbodylib.timesteps as ftim

#Simulation for trajectory
def simulate_trajectory(integrator, planets, adaptive_dt, dynamics):
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
    """ i=0

    print("Value of e:\t" + str(e))
    print(adaptive_dt.__name__)
    print(integrator.__name__)
    print(dynamics.__name__)
    print("Acceleration:\t" + str(planets.acc))
    print("Velocity module:\t" + str(planets.vel_mod()))
    print(velocity) """

    while t < 1000*Tperiod:
        """ print("####################################################")
        print(i)
        i+=1
        print("Value of e:\t" + str(e))
        print(adaptive_dt.__name__)
        print(integrator.__name__)
        print(dynamics.__name__)
        print("Acceleration:\t" + str(planets.acc))
        print("Velocity module:\t" + str(planets.vel_mod()))
        if planets.acc is None:
            print("inside program: None")
        else:
            print("inside program: Not None")
            print("Squared module of the acceleration:\t" + str(planets.acc_mod()))
            print("Norm of Acceleration:\t" + str(np.linalg.norm(planets.acc,axis=1)))
            print("Values of dt calculated:\t" + str(planets.vel_mod()/np.linalg.norm(planets.acc,axis=1)))
            print("Value of dt supposed to be chosen:\t" + str(np.nanmin(planets.vel_mod()/np.linalg.norm(planets.acc,axis=1)))) """
        tmin=0.001
        dt = adaptive_dt(planets,tmin=tmin)
        #print("Value of dt chosen:\t" + str(dt))
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
    """ i=0

    print("Value of e:\t" + str(e))
    print(adaptive_dt.__name__)
    print(integrator.__name__)
    print("Acceleration:\t" + str(planets.acc))
    print("Velocity module:\t" + str(planets.vel_mod()))
    print(velocity) """

    while t < 1000*Tperiod:
        """ print("####################################################")
        print(i)
        i+=1
        print("Value of e:\t" + str(e))
        print(adaptive_dt.__name__)
        print(integrator.__name__)
        print(dynamics.__name__)
        print("Acceleration:\t" + str(planets.acc))
        print("Velocity module:\t" + str(planets.vel_mod()))
        if planets.acc is None:
            print("inside program: None")
        else:
            print("inside program: Not None")
            print("Squared module of the acceleration:\t" + str(planets.acc_mod()))
            print("Norm of Acceleration:\t" + str(np.linalg.norm(planets.acc,axis=1)))
            print("Values of dt calculated:\t" + str(planets.vel_mod()/np.linalg.norm(planets.acc,axis=1)))
            print("Value of dt supposed to be chosen:\t" + str(np.nanmin(planets.vel_mod()/np.linalg.norm(planets.acc,axis=1)))) """
        tmin=0.001
        dt = adaptive_dt(planets,tmin=tmin)
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
adaptive_timestep = ftim.adaptive_timestep_vel_acc


for e in eccentricities:
    
    fig_ad, axes_ad = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), subplot_kw={'projection': '3d'})
    axes_ad = axes_ad.flatten()
    
    fig_tot_energy_ad, axes_tot_energy_ad = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes_tot_energy_ad = axes_tot_energy_ad.flatten()
    
    fig_error_ad,axes_loss_ad = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes_loss_ad = axes_loss_ad.flatten()

    planets = fic.ic_two_body(mass1=1., mass2=1., rp=2., e=e)
    
    for integrator, ax_ad, ax_en_ad,ax_loss_ad in zip(integrators, axes_ad, axes_tot_energy_ad,axes_loss_ad):
        if(integrator.__name__=="integrator_tsunami"):
            #tsunami call
            position, _, _, _, t, tot_en,error = simulate_trajectory_tsunami(fint.integrator_tsunami, planets.copy(), adaptive_timestep)
            plot_trajectory(position, ax_ad, f'{integrator.__name__}')
            plot_energy(tot_en,ax_en_ad, f'{integrator.__name__}')
            plot_energy_loss(error,ax_loss_ad,f'{integrator.__name__}')
        else:
            #other methods
            position, _, _, _, t, tot_en,error = simulate_trajectory(integrator, planets.copy(), adaptive_timestep, fdyn.acceleration_pyfalcon)
            plot_trajectory(position, ax_ad, f'{integrator.__name__}')
            plot_energy(tot_en,ax_en_ad, f'{integrator.__name__}')
            plot_energy_loss(error,ax_loss_ad,f'{integrator.__name__}')
            
    fig_ad.suptitle(f'Trajectories for e={e} with adaptive timestep "{adaptive_timestep.__name__}"', fontsize=16)
    plt.tight_layout()
    fig_ad.savefig(f"Trajectories_e_{e}_ad_{adaptive_timestep.__name__}.png")
    fig_tot_energy_ad.savefig(f"TotalEnergy_e_{e}_ad_{adaptive_timestep.__name__}.png")
    fig_error_ad.savefig(f"Energy_loss_e_{e}_ad_{adaptive_timestep.__name__}.png")

