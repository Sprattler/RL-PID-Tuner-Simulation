# Simulation code
import numpy as np
import pygame
import math
from scipy.integrate import solve_ivp
from config import *
from TDAgent import TD3Agent


# ------------------- Funktionen -------------------
def pendulum_dynamics(t: float, state: list, u_total: float) -> list:
    """Berechnet die Dynamik des Pendels."""
    theta, omega = state
    dtheta = omega
    domega = (-DAMPING * omega - (PENDULUM_MASS * GRAVITY * PENDULUM_LENGTH) * np.sin(theta) + u_total) / (
                PENDULUM_MASS * PENDULUM_LENGTH ** 2)
    return [dtheta, domega]


def pid_controller(theta: float, integral_error: float, last_error: float | None, dt: float) -> tuple:
    """PID-Regler zur Steuerung des Pendels."""
    error = SETPOINT - theta
    integral_error += error * dt
    derivative = (error - last_error) / dt if last_error is not None else 0.0
    u_pid = KP * error + KI * integral_error + KD * derivative
    u_pid = np.clip(u_pid, -MAX_TORQUE, MAX_TORQUE)
    return u_pid, integral_error, error


def calculate_external_torque(t: float) -> float:
    """Berechnet das externe sinusförmige Drehmoment."""
    return EXT_TORQUE_AMPLITUDE * np.sin(EXT_TORQUE_FREQUENCY * t)


def init_plot_data(max_points=200):
    """Initialisiert die Daten für die Live-Plots in Pygame."""
    return {
        'times': [],
        'errors': [],
        'integral_errors': [],
        'torques': [],
        'max_points': max_points
    }


def update_plot_data(plot_data, t, error, integral_error, torque):
    """Aktualisiert die Plot-Daten mit neuen Werten."""
    plot_data['times'].append(t)
    plot_data['errors'].append(error)
    plot_data['integral_errors'].append(integral_error)
    plot_data['torques'].append(torque)

    # Begrenze die Anzahl der Punkte
    if len(plot_data['times']) > plot_data['max_points']:
        plot_data['times'].pop(0)
        plot_data['errors'].pop(0)
        plot_data['integral_errors'].pop(0)
        plot_data['torques'].pop(0)


def draw_plots(screen, plot_data, font, plot_x=500, plot_y=100, plot_width=250, plot_height=100):
    """Zeichnet die Live-Plots in der Pygame-GUI mit Beschriftungen."""
    # Farben für die Plots
    ERROR_COLOR = (0, 0, 255)  # Blau
    INTEGRAL_ERROR_COLOR = (255, 0, 0)  # Rot
    TORQUE_COLOR = (0, 255, 0)  # Grün
    GRID_COLOR = (200, 200, 200)  # Grau
    BG_COLOR = (255, 255, 255)  # Weiß
    LABEL_COLOR = (0, 0, 0)  # Schwarz für Beschriftungen

    # Plot-Bereiche (drei übereinander)
    plot_areas = [
        (plot_x, plot_y, plot_width, plot_height),  # Error
        (plot_x, plot_y + plot_height + 20, plot_width, plot_height),  # Integral Error
        (plot_x, plot_y + 2 * (plot_height + 20), plot_width, plot_height)  # Torque
    ]

    # Beschriftungen
    labels = ["Error", "Integral Error", "Torque"]

    # Skalierungsparameter
    max_error = max(abs(min(plot_data['errors'], default=1)), abs(max(plot_data['errors'], default=1)), 1)
    max_integral = max(abs(min(plot_data['integral_errors'], default=1)),
                       abs(max(plot_data['integral_errors'], default=1)), 1)
    max_torque = MAX_TORQUE

    for i, (x, y, w, h) in enumerate(plot_areas):
        # Hintergrund zeichnen
        pygame.draw.rect(screen, BG_COLOR, (x, y, w, h))

        # Gitter zeichnen
        for gy in [y + h // 4, y + h // 2, y + 3 * h // 4]:
            pygame.draw.line(screen, GRID_COLOR, (x, gy), (x + w, gy))

        # Datenpunkte berechnen
        points = []
        for j in range(len(plot_data['times'])):
            t_norm = j / max(1, len(plot_data['times']) - 1) * w
            px = x + t_norm

            if i == 0:  # Error
                value = plot_data['errors'][j]
                py = y + h - (value / max_error * h / 2 + h / 2)
                color = ERROR_COLOR
            elif i == 1:  # Integral Error
                value = plot_data['integral_errors'][j]
                py = y + h - (value / max_integral * h / 2 + h / 2)
                color = INTEGRAL_ERROR_COLOR
            else:  # Torque
                value = plot_data['torques'][j]
                py = y + h - (value / max_torque * h / 2 + h / 2)
                color = TORQUE_COLOR

            points.append((px, py))

        # Linien zeichnen
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)

        # Rahmen zeichnen
        pygame.draw.rect(screen, (0, 0, 0), (x, y, w, h), 1)

        # Beschriftung zeichnen
        label_surface = font.render(labels[i], True, LABEL_COLOR)
        screen.blit(label_surface, (x - 80, y + h // 2 - 8))  # Links neben dem Plot, vertikal zentriert


def run_simulation(
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        font: pygame.font.Font,
        agent: TD3Agent,
        state: list,
        integral_error: float,
        last_error: float | None,
        t: float,
        pid_enabled: bool,
        episode: int,
        episode_reward: float,
        step: int,
        max_steps: int,
        plot_data: dict,
        dt: float = 0.01
) -> tuple[bool, list, float, float | None, float, bool, float, int, bool]:
    """Führt einen Simulationsschritt durch, entweder mit PID oder TD3-Agent."""

    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pid_enabled = not pid_enabled

    # Fehler berechnen
    error = SETPOINT - state[0]

    # Steuerung berechnen
    u_control = 0.0
    if pid_enabled:
        u_control, integral_error, last_error = pid_controller(state[0], integral_error, last_error, dt)
    else:
        integral_error += error * dt
        rl_state = np.array([error, integral_error])
        noise_scale = 0.3 if episode < 50 else 0.1
        u_control = agent.select_action(rl_state, noise_scale=noise_scale)[0]

    # Externes Drehmoment
    u_ext = calculate_external_torque(t)
    u_total = u_control + u_ext

    # Dynamik integrieren
    sol = solve_ivp(pendulum_dynamics, [t, t + dt], state, args=(u_total,), t_eval=[t + dt], method='RK45')
    next_state = sol.y[:, -1]
    t += dt

    # RL-Logik (nur wenn TD3 aktiv)
    done = False
    if not pid_enabled:
        next_error = SETPOINT - next_state[0]
        next_integral_error = integral_error + next_error * dt
        next_rl_state = np.array([next_error, next_integral_error])
        reward = agent.compute_reward(rl_state, u_control)
        episode_reward += reward

        done = step >= max_steps - 1
        agent.store_transition(rl_state, np.array([u_control]), reward, next_rl_state, done)
        if episode > 10:
            agent.train(iterations=1)

    # Bildschirm zeichnen
    screen.fill(BACKGROUND_COLOR)
    theta = next_state[0]
    bob_x = PIVOT[0] + SCALE * PENDULUM_LENGTH * np.sin(theta)
    bob_y = PIVOT[1] + SCALE * PENDULUM_LENGTH * np.cos(theta)

    pygame.draw.line(screen, PENDULUM_COLOR, PIVOT, (bob_x, bob_y), 2)
    pygame.draw.circle(screen, BOB_COLOR, (int(bob_x), int(bob_y)), 10)
    pygame.draw.circle(screen, BOB_OUTLINE_COLOR, (int(bob_x), int(bob_y)), 10, 1)
    pygame.draw.circle(screen, PIVOT_COLOR, PIVOT, 4)

    if pid_enabled:
        error_deg = math.degrees(error)
        texts = [
            f"Torque: {u_control:.2f} Nm",
            f"Ext. Torque: {u_ext:.2f} Nm",
            f"Error: {error_deg:.2f}°",
            f"Mode: PID"
        ]
    else:
        texts = [
            f"Episode: {episode}",
            f"Reward: {episode_reward:.2f}",
            f"Torque: {u_control:.2f} Nm",
            f"Ext. Torque: {u_ext:.2f} Nm"
        ]
    for i, text in enumerate(texts):
        surface = font.render(text, True, TEXT_COLOR)
        screen.blit(surface, (10, 10 + i * 20))

    # Plot-Daten aktualisieren und zeichnen
    update_plot_data(plot_data, t, error, integral_error, u_control)
    draw_plots(screen, plot_data, font)

    pygame.display.flip()
    clock.tick(FPS)

    return running, next_state, integral_error, last_error, t, pid_enabled, episode_reward, step + 1 if not pid_enabled else step, done


# ------------------- Initialisierung -------------------
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Pendulum with PID/TD3")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 16)

# TD3-Agent initialisieren
agent = TD3Agent(state_dim=2, action_dim=1, max_action=MAX_TORQUE)

# Plot-Daten initialisieren
plot_data = init_plot_data(max_points=200)

# Simulationsparameter
state = [0, 0.0]
integral_error = 0.0
last_error = None
t = 0.0
pid_enabled = USE_PID
episode = 0
episode_reward = 0
step = 0
max_steps = 200
dt = 0.01
episodes = 1000

# ------------------- Hauptschleife -------------------
running = True
while running and (pid_enabled or episode < episodes):
    running, state, integral_error, last_error, t, pid_enabled, episode_reward, step, done = run_simulation(
        screen, clock, font, agent, state, integral_error, last_error, t, pid_enabled, episode, episode_reward, step,
        max_steps, plot_data, dt
    )

    if not pid_enabled and (done or not running):
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")
        state = [0, 0.0]
        integral_error = 0.0  # Reset für TD3
        episode += 1
        episode_reward = 0
        step = 0
        plot_data = init_plot_data(max_points=200)  # Reset Plots für neue Episode

pygame.quit()