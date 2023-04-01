import gym
import time
import numpy as np
from environment import MyPendulumEnv
from ac import ACAgent
from dqn import DQNAgent

action_list = [np.array([-3.0]), np.array([0.0]), np.array([3.0])]

max_step = 500  # 1000 * 0.005 = 5s
# mode = "dqn"

mode = "ac"

if __name__ == "__main__":
    env = MyPendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if mode == "ac" else 3
    Agent = ACAgent if mode == "ac" else DQNAgent
    agent = Agent(env, state_dim, action_dim)
    agent.load_model("./res/model/")

    obs = env.reset()
    while True:
        for t in range(max_step):
            action = agent.get_action(obs)
            u = action if mode == "ac" else action_list[action]
            obs_next, reward, done, _ = env.step(u)
            agent.buffer.store(obs, action, reward, obs_next, done)

            env.render()
            time.sleep(0.01)
            obs = obs_next[:]

        obs = env.reset()










