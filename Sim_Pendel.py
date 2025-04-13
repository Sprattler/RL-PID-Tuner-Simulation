import numpy as np
import matplotlib.pyplot as plt
import math

# --------------------------------------------------------------
# Config
# --------------------------------------------------------------


Kp = 2
Ki = 2

TARGET_ANGLE = math.radians(90)
GRAVITY = 9.81
LENGTH = 1.0
MASS = 1.0
DAMPING = 0.1
MAX_TORQUE = 10.0
MIN_TORQUE = -10.0


I = MASS * (LENGTH ** 2)    # Trägheitsmoment Punktmasse


DT = 0.1
T_total = 20.0  # dauer der Simulation
N_steps = int(T_total / DT)


# --------------------------------------------------------------
# Initialisierung
# --------------------------------------------------------------
theta = 0.0
theta_dot = 0.0
integral = 0.0

time_log = []
theta_log = []
theta_dot_log = []
control_log = []
error_log = []

# --------------------------------------------------------------
# Simulation
# --------------------------------------------------------------
for step in range(N_steps):
    t = step * DT

    # Regelungsfehler berechnen:
    error = TARGET_ANGLE - theta
    integral += error * DT
    u = Kp * error + Ki * integral
    u = np.clip(u, MIN_TORQUE, MAX_TORQUE)

    # Dynamik des Pendels
    theta_dot_new = theta_dot + DT * (- (DAMPING / I) * theta_dot - (GRAVITY / LENGTH) * np.sin(theta) + (1.0 / I) * u)
    theta_new = theta + DT * theta_dot_new

    # Speichern
    time_log.append(t)
    theta_log.append(theta_new)
    theta_dot_log.append(theta_dot_new)
    control_log.append(u)
    error_log.append(error)

    # Update für den nächsten Schritt
    theta = theta_new
    theta_dot = theta_dot_new

# --------------------------------------------------------------
# Plots: Darstellung der Ergebnisse
# --------------------------------------------------------------
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time_log, theta_log, label="Winkel θ (rad)")
plt.axhline(y=TARGET_ANGLE, color='r', linestyle='--', label="Sollwinkel")
plt.ylabel("Winkel (rad)")
plt.title("PI-reguliertes Pendel")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
