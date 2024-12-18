import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
from extracting_features import *

gym.register_envs(ale_py)

# Initialise the environment
env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    visualize_frame(process_frame(observation))

env.close()