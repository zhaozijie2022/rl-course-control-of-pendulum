import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
import numpy as np

import gym


class MLP(nn.Cell):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, hidden_dim)
        self.fc3 = nn.Dense(hidden_dim, output_dim)

    def construct(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


class Buffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.state = np.zeros([max_size, state_dim])
        self.action = np.zeros([max_size, 1])
        self.reward = np.zeros([max_size, 1])
        self.next_state = np.zeros([max_size, state_dim])
        self.done = np.zeros([max_size, 1])
        self.ptr = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = self.next_state[idx]
        done = self.done[idx]
        return state, action, reward, next_state, done


class DQNAgent:
    def __init__(self, env, state_dim, action_dim, gamma=0.98, lr=0.001, batch_size=256, max_size=100000):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.max_size = max_size
        self.buffer = Buffer(state_dim=self.state_dim, action_dim=self.action_dim, max_size=self.max_size)
        self.q = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.q.set_train(True)
        self.q_target = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.q_target.set_train(False)

        self.loss_fn = nn.MSELoss()
        self.optimizer = nn.Adam(self.q.trainable_params(), lr)

        self.tau = 0.01

    def get_action(self, state):
        state = ms.Tensor(state, ms.float32).view(1, -1)  # ms中要求列向量
        q = self.q.construct(state)
        action = np.argmax(q.asnumpy())
        return action

    def update_target(self):
        for p, p_target in zip(self.q.trainable_params(), self.q_target.trainable_params()):
            p_target.set_data(p_target * (1 - self.tau) + p * self.tau)

    def forward_fn(self, data, label):
        # 以[state, action]为data, y为label, q就是logits
        state, action = data
        q = self.q.construct(state)
        logits = ms.numpy.take_along_axis(q, action, axis=1)  # 索引方法来自chatGPT
        loss = self.loss_fn(logits, label)
        return loss, logits

    def train_step(self, data, label):
        grad_fn = ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def train(self):
        if self.buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = ms.Tensor(state, ms.float32).view(self.batch_size, -1)
        action = ms.Tensor(action, ms.int32).view(self.batch_size, -1)
        reward = ms.Tensor(reward, ms.float32).view(self.batch_size, -1)
        next_state = ms.Tensor(next_state, ms.float32).view(self.batch_size, -1)
        done = ms.Tensor(done, ms.float32).view(self.batch_size, -1)

        data = [state, action]
        q_target = self.q_target(next_state)
        q_target = ms.numpy.max(q_target, axis=1).reshape(self.batch_size, -1)
        y = reward + self.gamma * (1 - done) * q_target

        self.train_step(data, y)

        self.update_target()