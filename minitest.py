from curling import CurlingEnv
import numpy as np

env = CurlingEnv()
env.reset()


for t in range(200):
    env.step(-5.0)
    print(env.state)








