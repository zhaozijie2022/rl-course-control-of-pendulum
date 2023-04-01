# import mindspore as ms
# import mindspore.nn as nn
# from mindspore import ops

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pickle

import gym
import numpy as np

from ac import MLP
from ac import Buffer

class QBuffer(Buffer):
    def __init__(self, state_dim, action_dim, max_size=100000):
        super(QBuffer, self).__init__(state_dim, action_dim, max_size)
        self.action = np.zeros([max_size, 1])

class DQNAgent:
    def __init__(self, env, state_dim, action_dim, gamma=0.98, lr=0.001, batch_size=256, max_size=100000):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 0.05
        self.batch_size = batch_size
        self.max_size = max_size
        self.buffer = QBuffer(state_dim=self.state_dim, action_dim=self.action_dim, max_size=self.max_size)
        self.q = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.q_target = MLP(input_dim=self.state_dim, output_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr)

        self.tau = 0.01

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        q = self.q.forward(state)
        action = torch.argmax(q)
        return action.numpy()

    def update_target(self):
        for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def train(self):
        if self.buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32).view(self.batch_size, -1)
        action = torch.tensor(action, dtype=torch.int64).view(self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float32).view(self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(self.batch_size, -1)
        done = torch.tensor(done, dtype=torch.float32).view(self.batch_size, -1)

        q = self.q.forward(state)
        q = torch.gather(q, dim=1, index=action)
        q_target = self.q_target(next_state)
        q_target = torch.max(q_target, dim=1)[0].view(self.batch_size, -1)
        y = reward + self.gamma * (1 - done) * q_target

        loss = F.mse_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()

    def save_model(self, path):
        path = os.path.join(path, "q_net.pth")
        torch.save(self.q.state_dict(), path)

    def load_model(self, path):
        path = os.path.join(path, "q_net.pth")
        self.q.load_state_dict(torch.load(path))
        self.q_target.load_state_dict(self.q.state_dict())



