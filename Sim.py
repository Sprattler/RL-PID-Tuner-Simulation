import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.hierarchy import flow_hierarchy

# ---------------------------------------------
# PI Parameter
# ---------------------------------------------
Kp = 4.0312
Ki = 1.3579

# ---------------------------------------------
# System-Parameter
# ---------------------------------------------
h_target = 1.0          # Zielwasserstand
flow    = 10            # Max Wasserfluss
leak_rate = 0.5         # Leckrate
dt = 0.1                # Zeitschritt
T = 10                  # Gesamtdauer
N = int(T / dt)         # Sim-Schritte


h = 0.0
integral = 0.0
time = []
water_level = []
control_input = []
error_list = []
errorI_list = []

# ---------------------------------------------
# Simulation
# ---------------------------------------------
for step in range(N):
    t = step * dt
    error = h_target - h
    integral += error * dt
    u = Kp * error + Ki * integral
    u = np.clip(u, 0.0, flow)

    dh = dt * (u - leak_rate * h)
    h += dh

    # Logging
    time.append(t)
    water_level.append(h)
    control_input.append(u)
    error_list.append(error)
    errorI_list.append(integral)

# ---------------------------------------------
# Plots
# ---------------------------------------------
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, water_level, label='Wasserstand h(t)')
plt.axhline(y=h_target, color='r', linestyle='--', label='Ziel')
plt.ylabel("h(t) [m]")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, errorI_list, label='error_integrated', color='green')
plt.ylabel("u(t)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
