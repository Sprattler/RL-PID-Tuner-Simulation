import math
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------------
# Konfiguration ("Defines")
# --------------------------------------------------------------

TARGET_LEVEL = math.radians(90)
GRAVITY = 9.81
LENGTH = 1.0
MASS = 1.0
DAMPING = 0.1
MAX_ACTION = 10.0
MIN_ACTION = -10.0
I = MASS * (LENGTH ** 2)

EPISODES = 100
MAX_STEPS = 500
DT = 0.01

ACTOR_LR = 1e-3     # Lernrate Actor
CRITIC_LR = 1e-3    # Lernrate Critic
DISCOUNT = 0.99     # Gamma
TAU = 0.005         # Für soft updates der Targets

POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2

EXPLORATION_NOISE = 0.1
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 1e6

# -------------------- Umgebung: Pendelumgebung --------------------
class PendulumEnv(gym.Env):

    def __init__(self):
        super(PendulumEnv, self).__init__()
        self.dt = DT
        self.target = TARGET_LEVEL
        self.max_steps = MAX_STEPS
        self.g = GRAVITY
        self.L = LENGTH
        self.I = MASS * (LENGTH ** 2)
        self.damping = DAMPING

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=MIN_ACTION, high=MAX_ACTION, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.theta = 0.0  # Startwinkel (Radiant)
        self.theta_dot = 0.0  # Startwinkelgeschwindigkeit
        self.integral = 0.0
        error = self.target - self.theta
        return np.array([error, self.integral], dtype=np.float32)

    def step(self, action):
        u = action[0]
        # Diskrete Dynamik:
        theta_dot_new = self.theta_dot + self.dt * (
                - (self.damping / self.I) * self.theta_dot
                - (self.g / self.L) * np.sin(self.theta)
                + (1.0 / self.I) * u
        )
        theta_new = self.theta + self.dt * theta_dot_new

        self.theta = theta_new
        self.theta_dot = theta_dot_new
        self.current_step += 1

        error = self.target - self.theta
        self.integral += error * self.dt
        observation = np.array([error, self.integral], dtype=np.float32)
        reward = - (error ** 2)  # Negativer quadratischer Fehler
        done = (self.current_step >= self.max_steps)
        info = {'theta': self.theta, 'u': u}
        return observation, reward, done, info

    def render(self, mode='human'):
        pass


# ------------ Benutzerdefinierter PI-Layer (analog zu MATLAB) ------------
class FullyConnectedPILayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedPILayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(output_size, input_size))

    def forward(self, x):
        weight = torch.abs(self.weights)
        return torch.nn.functional.linear(x, weight, bias=None)


# -------------------------- TD3 Actor --------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.pi_layer = FullyConnectedPILayer(input_size=state_dim, output_size=action_dim)

    def forward(self, state):
        return self.pi_layer(state)


# ---------------------- TD3 Critic-Netzwerk ----------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q1 = nn.Linear(64, 1)

        self.fc1_2 = nn.Linear(state_dim + action_dim, 64)
        self.fc2_2 = nn.Linear(64, 64)
        self.q2 = nn.Linear(64, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = torch.relu(self.fc1(xu))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.q1(x1)
        x2 = torch.relu(self.fc1_2(xu))
        x2 = torch.relu(self.fc2_2(x2))
        q2 = self.q2(x2)
        return q1, q2

    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = torch.relu(self.fc1(xu))
        x1 = torch.relu(self.fc2(x1))
        return self.q1(x1)


# ---------------------------- Replay Buffer ----------------------------
class ReplayBuffer(object):
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            states.append(np.asarray(state))
            actions.append(np.asarray(action))
            rewards.append(np.asarray(reward))
            next_states.append(np.asarray(next_state))
            dones.append(np.asarray(done))
        return (
            torch.FloatTensor(np.asarray(states)),
            torch.FloatTensor(np.asarray(actions)),
            torch.FloatTensor(np.asarray(rewards)).unsqueeze(1),
            torch.FloatTensor(np.asarray(next_states)),
            torch.FloatTensor(np.asarray(dones)).unsqueeze(1)
        )


# ------------------------------ TD3-Agent ------------------------------
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=BATCH_SIZE, discount=DISCOUNT,
              tau=TAU, policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP, policy_freq=POLICY_FREQ):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(MIN_ACTION, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# ----------------------------- Main Loop -----------------------------
def main(episodes):
    env = PendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    td3_agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = td3_agent.select_action(state)
            noise = np.random.normal(0, max_action * EXPLORATION_NOISE, size=action_dim)
            action = (action + noise).clip(MIN_ACTION, max_action)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += reward
            if len(replay_buffer.storage) > 1000:
                td3_agent.train(replay_buffer)
        print(f"Episode {episode + 1:03d}: Reward={str(round(episode_reward, 2))}, Winkel={str(round(math.degrees(info['theta']), 5))}°")

    torch.save(td3_agent.actor.state_dict(), "td3_actor.pth")
    print("Modell gespeichert als td3_actor.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()
    main(args.episodes)
