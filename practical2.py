import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate  import solve_ivp

def system_two_odes(t, y):
    x1, x2 = y
    dx1dt = -3 * x1 - x2
    dx2dt = x1
    return [dx1dt, dx2dt]
time_span = (0,10 )
# system of two ODEs 
initial_conditions = [1, 0]
t_eval = np.arange(*time_span, 0.01)
sol = solve_ivp(system_two_odes, time_span, initial_conditions, t_eval=t_eval)
plt.plot(sol.t, sol.y[0], label='x1(t)')
plt.plot(sol.t, sol.y[1], label='x2')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Solution of system of two ODEs')
plt.legend()
plt.show()

# (b) Second Order ODE
def second_order_ode(t, y):
    y, dy = y
    d2ydt2 = -9 * dy - 4 * y + np.exp(-2 * t) * np.sin(t)
    return [dy, d2ydt2]

t_span = (0, 10)
y0 = [1, 5]
t_eval = np.arange(0, 10.1, 0.1)
sol = solve_ivp(second_order_ode, t_span, y0, t_eval=t_eval)

plt.figure()
plt.plot(sol.t, sol.y[0], label='y')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Solution of second order ODE')
plt.show()


# (c) System of Three ODEs
def system_three_odes(t, y):
    x, y, z = y
    dxdt = 5 * x + 2 * y - z
    dydt = x + 8 * z
    dzdt = 6 * x - 3 * y + 5 * z
    return [dxdt, dydt, dzdt]

t_span = (0, 0.5)
y0 = [1, 2, 3]
t_eval = np.arange(0, 0.51, 0.01)
sol = solve_ivp(system_three_odes, t_span, y0, t_eval=t_eval)

plt.figure()
plt.plot(sol.t, sol.y[0], label='x')
plt.plot(sol.t, sol.y[1], label='y')
plt.plot(sol.t, sol.y[2], label='z')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Solution of system of three ODEs')
plt.legend()
plt.show()

