import gym
import time
import numpy as np
from environment import MyPendulumEnv
from ac import ACAgent
from dqn import DQNAgent
from torch.utils.tensorboard import SummaryWriter

action_list = [np.array([-3.0]), np.array([0.0]), np.array([3.0])]

max_step = 500  # 1000 * 0.005 = 5s
test_rate = 50
target_rew = -6000
max_eps = 150
mode = "dqn"

# mode = "ac"

if __name__ == "__main__":
    env = MyPendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if mode == "ac" else 3
    Agent = ACAgent if mode == "ac" else DQNAgent
    agent = Agent(env, state_dim, action_dim)
    writer = SummaryWriter("./logger")

    obs = env.reset()
    rew_eps = [0.0]
    theta_eps = [0.0]
    ep = 0
    while ep < max_eps:
        for t in range(max_step):
            action = agent.get_action(obs)
            u = action if mode == "ac" else action_list[action]
            obs_next, reward, done, _ = env.step(u)
            agent.buffer.store(obs, action, reward, obs_next, done)

            obs = obs_next[:]
            rew_eps[-1] += reward
            theta_eps[-1] += np.abs(env.state[0])

            agent.train()

        theta_eps[-1] /= max_step
        print("Episode: %d, Reward: %.2f" % (ep, rew_eps[-1]))
        ep += 1
        obs = env.reset()
        writer.add_scalar("reward", rew_eps[-1], ep)
        writer.add_scalar("theta", theta_eps[-1], ep)
        rew_eps.append(0.0)
        theta_eps.append(0.0)

        if ep % test_rate == 0:
            for tt in range(max_step):
                action = agent.get_action(obs)
                u = action if mode == "ac" else action_list[action]
                obs_next, reward, done, _ = env.step(u)
                env.render()
                time.sleep(0.01)
                obs = obs_next[:]
                rew_eps[-1] += reward

    env.close()
    agent.save_model("./res/model/")
    # agent.save_buffer("./res/buffer/")








