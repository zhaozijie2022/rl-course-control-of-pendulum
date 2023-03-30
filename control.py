import gym
import time
import numpy as np
from curling import MyPendulumEnv
from agent import Agent
from torch.utils.tensorboard import SummaryWriter

max_step = 500  # 1000 * 0.005 = 5s

if __name__ == "__main__":
    # display_env = gym.make("Pendulum-v0")
    # display_env.reset()
    env = MyPendulumEnv()
    # env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Agent(env, state_dim=state_dim, action_dim=action_dim)
    s = env.reset()

    writer = SummaryWriter("./logger")

    t = 0
    rew_eps = [0.0]
    ep = 0
    while ep < 10000:
        t += 1
        u = agent.get_action(s)
        s_next, r, d, _ = env.step(u)
        agent.buffer.store(s, u, r, s_next, d)

        s = s_next[:]
        rew_eps[-1] += r
        agent.train()
        if d or t == max_step:
            print("Episode: %d, Reward: %.2f" % (ep, rew_eps[-1]))
            t = 0
            ep += 1
            s = env.reset()
            writer.add_scalar("reward", rew_eps[-1], ep)
            rew_eps.append(0.0)
        if ep % 50 == 0:
            # display_env.env.state = env.state
            # print(u, env.state[1])
            env.render()
            time.sleep(0.05)











