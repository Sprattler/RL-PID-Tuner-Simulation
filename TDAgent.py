import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# ------------------- Konstanten (Defines) -------------------
# Netzwerk-Parameter
STATE_DIM = 2  # Dimension des Zustands (Fehler, integraler Fehler)
ACTION_DIM = 1  # Dimension der Aktion (Drehmoment)
MAX_ACTION = 20  # Maximales Drehmoment (entspricht MAX_TORQUE aus config.py)
HIDDEN_SIZE = 32  # Hidden Layer Critics

# Lern-Parameter
LR_ACTOR = 1e-3  # Lernrate des Actors
LR_CRITIC = 1e-3  # Lernrate der Critics
GAMMA = 0.99  # Discount-Faktor
TAU = 0.005  # Soft-Update-Parameter für Target-Netzwerke
POLICY_NOISE = 0.2  # Rauschen für Target-Policy-Smoothing
NOISE_CLIP = 0.5  # Begrenzung des Rauschens
POLICY_FREQ = 2  # Frequenz der Actor-Updates (verzögerter Update)

# Replay Buffer
BUFFER_SIZE = 1000000  # Maximale Größe des Replay Buffers
BATCH_SIZE = 128  # Batch-Größe für das Training


# ------------------- Critic-Netzwerk -------------------
class CriticNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super(CriticNetwork, self).__init__()
        # State Path
        self.state_fc1 = nn.Linear(state_dim, HIDDEN_SIZE)

        # Action Path
        self.action_fc1 = nn.Linear(action_dim, HIDDEN_SIZE)

        # Common Path
        self.concat_fc1 = nn.Linear(HIDDEN_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)
        self.output_fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.state_fc1(state))
        action_out = torch.relu(self.action_fc1(action))
        concat = torch.cat([state_out, action_out], dim=-1)
        x = torch.relu(self.concat_fc1(concat))
        q_value = self.output_fc(x)
        return q_value


# ------------------- Actor-Netzwerk -------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim, bias=False)
        self.max_action = max_action

    def forward(self, state):
        action = torch.tanh(self.fc(state)) * self.max_action
        return action


# ------------------- TD3-Agent -------------------
class TD3Agent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, max_action=MAX_ACTION):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Netzwerke
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_1_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_2_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimierer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=LR_CRITIC)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=LR_CRITIC)

        # Replay Buffer
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.step_counter = 0

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale * self.max_action, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def compute_reward(self, state, action):
        """
        Belohnungsfunktion: -(error^2 + 0.01 * u^2)
        state[0]: Fehler (error)
        action: Aktion (u)
        """
        error = state[0]
        action_penalty = 0.01 * action ** 2
        return -(error ** 2 + action_penalty)

    def train(self, iterations=1):
        for _ in range(iterations):
            if len(self.replay_buffer) < BATCH_SIZE:
                return

            # Sample Batch
            batch = random.sample(self.replay_buffer, BATCH_SIZE)
            state, action, reward, next_state, done = zip(*batch)

            state = torch.FloatTensor(np.array(state)).to(self.device)
            action = torch.FloatTensor(np.array(action)).to(self.device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)

            # Target Action mit Noise
            noise = torch.normal(0, POLICY_NOISE, size=action.shape).to(self.device)
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Target Q-Werte
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * GAMMA * target_q

            # Critic Update
            current_q1 = self.critic_1(state, action)
            current_q2 = self.critic_2(state, action)

            critic_1_loss = nn.MSELoss()(current_q1, target_q.detach())
            critic_2_loss = nn.MSELoss()(current_q2, target_q.detach())

            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # Actor Update (verzögert)
            self.step_counter += 1
            if self.step_counter % POLICY_FREQ == 0:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Target Netzwerke aktualisieren
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)