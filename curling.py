import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np


class MyPendulumEnv(PendulumEnv):
    def __init__(self):
        super().__init__()
        self.m = 0.055
        self.g = 9.81
        self.l = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5

        self.dt = 0.005
        self.max_speed = 15 * np.pi
        self.max_torque = 3.0

        high = np.array([1, 1, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)
        super().seed()
        self.state = np.array([np.pi, 0.0])

    def reset(self):
        self.state = np.array([np.pi, 0.0])
        self.last_u = None
        return self._get_obs()

    def reward(self, action):
        theta, omega = self.state
        return -5 * theta ** 2 - 0.1 * omega ** 2 - action ** 2

    def step(self, u):
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        theta, omega = self.state
        omega_dot = (self.m * self.g * self.l * np.sin(theta)
                     - self.b * omega
                     - self.K ** 2 * omega / self.R
                     + self.K * u / self.R) / self.J

        omega = np.clip(omega + omega_dot * self.dt, -15 * np.pi, 15 * np.pi)
        theta = theta + omega * self.dt
        if theta < - np.pi:
            theta += 2 * np.pi
        elif theta >= np.pi:
            theta -= 2 * np.pi
        self.state = [theta, omega]
        # self.state = np.array([theta + omega * self.dt, omega + omega_dot * self.dt])
        return self._get_obs(), self.reward(u), False, {}


