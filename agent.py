import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dim)

    def forward(self, x):
        x = self.mlp(x)
        return torch.tanh(x) * 3


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.mlp = MLP(input_dim, output_dim, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        value = self.mlp(x)
        return value


class Buffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.state = np.zeros([max_size, state_dim])
        self.action = np.zeros([max_size, action_dim])
        self.reward = np.zeros([max_size, 1])
        self.next_state = np.zeros([max_size, state_dim])
        self.done = np.zeros([max_size, 1])
        self.ptr = 0

    def store(self, state, action, reward, next_state, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.state), batch_size)
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = self.next_state[idx]
        done = self.done[idx]
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.state)


class Agent:
    def __init__(self, env, state_dim, action_dim, gamma=0.98, lr=0.001, batch_size=256, max_size=100000):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.max_size = max_size
        self.buffer = Buffer(state_dim=state_dim, action_dim=action_dim, max_size=max_size)
        self.actor = Actor(input_dim=state_dim, output_dim=action_dim)
        self.critic = Critic(input_dim=state_dim + action_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_actor = Actor(input_dim=state_dim, output_dim=action_dim)
        self.target_critic = Critic(input_dim=state_dim + action_dim)
        self.tau = 0.01

    def get_action(self, state):
        if np.random.random() < 0.05:
            return np.random.uniform(-3, 3, size=1)
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor.forward(state)
        return action.detach().numpy()

    def update_target(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32).view(self.batch_size, -1)
        action = torch.tensor(action, dtype=torch.float32).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float32).view(self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(self.batch_size, -1)
        done = torch.tensor(done, dtype=torch.float32).view(self.batch_size, -1)

        next_action = self.target_actor.forward(next_state)
        next_value = self.target_critic.forward(next_state, next_action)
        value = self.critic.forward(state, action)
        td_target = reward + self.gamma * next_value * (1 - done)

        critic_loss = F.mse_loss(value, td_target.detach())
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, action).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.update_target()
