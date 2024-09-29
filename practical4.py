import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import lsim, StateSpace

# Constants for MSD
M = 6
B = 5
K = 3
F = 1

# Constants for RLC
R = 1
L = 1
C = 1

# Time vector
t = np.linspace(0, 15, 1000)

# Function for Mass-Spring-Damper (State-Space Representation)
def msd_state_space(x, t, A, B, u):
    return A @ x + B * u

# MSD State Space Matrices
A_msd = np.array([[0, 1], [-K/M, -B/M]])
B_msd = np.array([0, 1/M])
C_msd = np.array([1, 0])
D_msd = 0
u_msd = F  # Force input

# Initial conditions
x0_msd = [0, 0]  # Initial displacement and velocity

# Solving the system using odeint
x_msd = odeint(msd_state_space, x0_msd, t, args=(A_msd, B_msd, u_msd))
displacement_msd = x_msd[:, 0]

# RLC Circuit (State-Space Representation)
def rlc_state_space(x, t, A, B, u):
    return A @ x + B * u

# RLC State Space Matrices
A_rlc = np.array([[-R/L, -1/L], [1/C, 0]])
B_rlc = np.array([1/L, 0])
C_rlc = np.array([0, 1])
D_rlc = 0
u_rlc = 1  # Step input voltage

# Initial conditions
x0_rlc = [0, 0]  # Initial current and voltage

# Solving the system using odeint
x_rlc = odeint(rlc_state_space, x0_rlc, t, args=(A_rlc, B_rlc, u_rlc))
voltage_rlc = x_rlc[:, 1]

# Plotting the results
plt.figure(figsize=(10, 8))

# Plot MSD displacement (State-Space)
plt.subplot(221)
plt.plot(t, displacement_msd)
plt.title("MSD by State Space Block")
plt.xlabel('Time (s)')
plt.ylabel('Displacement')

# Plot MSD displacement and velocity (Block Diagram)
plt.subplot(222)
plt.plot(t, x_msd[:, 0], label='Displacement (x1)')
plt.plot(t, x_msd[:, 1], label='Velocity (x2)')
plt.title("MSD by Block Diagram")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot RLC voltage (State-Space)
plt.subplot(223)
plt.plot(t, voltage_rlc)
plt.title("RLC by State Space Block")
plt.xlabel('Time (s)')
plt.ylabel('Capacitor Voltage')

# Plot RLC current and voltage (Block Diagram)
plt.subplot(224)
plt.plot(t, x_rlc[:, 0], label='Loop Current (x1)')
plt.plot(t, voltage_rlc, label='Capacitor Voltage (x2)')
plt.title("RLC by Block Diagram")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
