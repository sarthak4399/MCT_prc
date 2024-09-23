import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, step
from sympy import symbols, Eq, dsolve, Function, Derivative

# Q1. Mechanical System
B = 5
M = 6
K = 3
F = 1  # unit step input

# (a) Using differential equations
t = symbols('t')
x1 = Function('x1')(t)
x2 = Function('x2')(t)

stateqn1 = Eq(Derivative(x1, t), x2)
stateqn2 = Eq(Derivative(x2, t), -(K/M)*x1 - (B/M)*x2 + (F/M))

sol = dsolve([stateqn1, stateqn2], ics={x1.subs(t, 0): 0, x2.subs(t, 0): 0})
x1_sol = sol[0].rhs
x2_sol = sol[1].rhs

# Time array for plotting
time = np.linspace(0, 20, 2000)

# Evaluate the solutions
x1_func = [x1_sol.subs(t, i).evalf() for i in time]
x2_func = [x2_sol.subs(t, i).evalf() for i in time]

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.plot(time, x1_func, label='Displacement (x1)')
plt.plot(time, x2_func, label='Velocity (x2)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('MSD by solving ODEs')

# (b) Using State Space matrix representation
A = np.array([[0, 1], [-K/M, -B/M]])
B = np.array([[0], [F/M]])
C = np.array([[1, 0]])
D = np.array([[0]])

system = StateSpace(A, B, C, D)
time, response = step(system, T=time)

plt.subplot(222)
plt.plot(time, response)
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title('MSD Step Response from State-Space')

# Q2. Electrical System
R = 1
L = 1
C = 1
Vin = 1  # unit step input

# (a) Using differential equations
x1 = Function('x1')(t)
x2 = Function('x2')(t)

stateqn1 = Eq(Derivative(x1, t), -(R/L)*x1 - (1/L)*x2 + (Vin/L))
stateqn2 = Eq(Derivative(x2, t), (1/C)*x1)

sol = dsolve([stateqn1, stateqn2], ics={x1.subs(t, 0): 0, x2.subs(t, 0): 0})
x1_sol = sol[0].rhs
x2_sol = sol[1].rhs

# Evaluate the solutions
x1_func = [x1_sol.subs(t, i).evalf() for i in time]
x2_func = [x2_sol.subs(t, i).evalf() for i in time]

# Plot
plt.subplot(223)
plt.plot(time, x1_func, label='Loop Current (x1)')
plt.plot(time, x2_func, label='Capacitor Voltage (x2)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('RLC by solving ODEs')

# (b) Using State Space matrix representation
A = np.array([[-R/L, -1/L], [1/C, 0]])
B = np.array([[1/L], [0]])
C = np.array([[0, 1]])
D = np.array([[0]])

system = StateSpace(A, B, C, D)
time, response = step(system, T=time)

plt.subplot(224)
plt.plot(time, response)
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('RLC Step Response from State-Space')

plt.tight_layout()
plt.show()
