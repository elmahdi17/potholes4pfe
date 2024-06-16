import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
m = 250  # mass of the car body (kg)
k = 16000  # suspension stiffness (N/m)
c = 1000  # damping coefficient (N s/m)

# PID Controller gains
Kp = 200   # Proportional gain (reduced for smoother response)
Ki = 10     # Integral gain (increased for better steady-state response)
Kd = 50    # Derivative gain (increased for improved transient re

# Function to simulate the road disturbance with potholes
def road_disturbance(t, pothole):
    # Initialize road profile as a simple sinusoidal function
    road_profile = 0.1 * np.sin(2 * np.pi * 0.5 * t)

    # Add pothole to the road profile
    pothole_time = pothole['time']
    pothole_depth = pothole['depth']
    pothole_width = pothole['width']
    
    road_profile += (pothole_depth *
                     np.exp(-((t - pothole_time) ** 2) / (2 * pothole_width ** 2)))
    
    return road_profile

# System dynamics
def suspension_system(state, t, m, k, c, Kp, Ki, Kd, pothole):
    # State variables
    x, v, xi = state

    # Road disturbance
    road = road_disturbance(t, pothole)

    # Control force from the actuator using PID control
    error = road - x
    control_force = Kp * error + Ki * xi + Kd * (-v)
    
    # Equations of motion
    dxdt = v
    dvdt = (control_force - c * v - k * x) / m
    dxidt = error  # Integral of error
    
    return [dxdt, dvdt, dxidt]

# Function for suspension simulation
def suspension_simulation_code(pothole):
    # Initial conditions
    x0 = 0  # Initial displacement
    v0 = 0  # Initial velocity
    xi0 = 0  # Initial integral of error

    initial_state = [x0, v0, xi0]

    # Time vector
    t = np.linspace(0, 10, 1000)  # 10 seconds of simulation

    # Function to update plot data
    def update_plot(frame):
        time = t[:frame+1]
        road_profile = road_disturbance(time, pothole)
        solution = odeint(suspension_system, initial_state, time, args=(m, k, c, Kp, Ki, Kd, pothole))
        x = solution[:, 0]
        v = solution[:, 1]

        plt.cla()  # Clear previous plot
        plt.plot(time, road_profile, 'r-', label='Road Profile')
        plt.plot(time, v, label='Suspension Reaction')
        plt.xlabel('Time (s)')
        plt.ylabel('active suspension reaction')
        plt.title('Potholes Detection')
        plt.legend()

    # Animate the plot
    fig = plt.figure(figsize=(12, 8))
    ani = animation.FuncAnimation(fig, update_plot, frames=len(t), interval=10)

    plt.tight_layout()
    plt.show()

# Test the suspension simulation with one pothole
pothole = {'time': 0.5, 'depth': -1, 'width': 0.08}
suspension_simulation_code(pothole)
