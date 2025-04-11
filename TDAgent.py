# TDAgent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from config import MAX_TORQUE

# Hyperparameter
STATE_DIM = 2  # [error, integral_error]
ACTION_DIM = 1  # Drehmoment
GAMMA = 0.99  # Discount-Faktor
TAU = 0.005  # Soft-Update Parameter
LR_ACTOR = 0.0001  # Lernrate Actor
LR_CRITIC = 0.001  # Lernrate Critic
BATCH_SIZE = 64
MEMORY_SIZE = 100000
NOISE_CLIP = 0.2
POLICY_NOISE = 0.1
POLICY_FREQ = 2
COST_FACTOR = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# Actor-Netzwerk
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim, bias=False)  # Kein Bias: u = [∫e, e] * [Ki, Kp]^T
        self.max_action = max_action

    def forward(self, state):
        # Gewichte auf positive Werte beschränken (abs)
        with torch.no_grad():
            self.fc.weight.data.abs_()
        u = self.fc(state)
        return torch.tanh(u) * self.max_action


# Critic-Netzwerk
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 Netzwerk
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        # Q2 Netzwerk
        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.fc1(sa))
        q1 = self.fc2(q1)
        q2 = torch.relu(self.fc3(sa))
        q2 = self.fc4(q2)
        return q1, q2


# TD3-Agent
class TD3Agent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_TORQUE):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.step_count = 0

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale * self.max_action, size=ACTION_DIM)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def compute_reward(self, state, action, Q=1.0, R=COST_FACTOR):
        """
        Belohnungsfunktion: Negativer LQG-Kostenwert
        """
        error, integral_error = state
        return -(Q * (error ** 2) + R * action ** 2)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train(self, iterations=1):
        if len(self.memory) < BATCH_SIZE:
            return

        for _ in range(iterations):
            self.step_count += 1
            states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)

            # Zielaktionen mit Rauschen
            noise = torch.clamp(
                torch.normal(mean=0., std=POLICY_NOISE, size=actions.shape).to(device),
                -NOISE_CLIP, NOISE_CLIP
            )
            next_actions = torch.clamp(
                self.actor_target(next_states) + noise, -self.max_action, self.max_action
            )

            # Ziel-Q-Werte
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q

            # Aktuelle Q-Werte
            current_Q1, current_Q2 = self.critic(states, actions)

            # Critic-Verlust
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor-Update (verzögert)
            if self.step_count % POLICY_FREQ == 0:
                actor_loss = -self.critic(states, self.actor(states))[0].mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft-Update der Zielnetzwerke
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)