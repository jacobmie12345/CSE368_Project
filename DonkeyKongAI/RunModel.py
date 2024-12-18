import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
from DonkeyKongModel import DuelingDQN
from extracting_features import stack_frames


def play_game(env,path,scale):
    height = int(210*scale)
    width = int(160*scale)
    action_space = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingDQN(action_space,height,width).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    state=env.reset()[0]
    frame_stack = deque(maxlen=4)
    stacked_state = stack_frames(state, frame_stack, scale=scale)
    stacked_state = np.expand_dims(stacked_state, axis=0)
    stacked_state = torch.FloatTensor(stacked_state).to(device)

    manual_start = True
    lives = 2
    done = False
    while not done:
        with torch.no_grad():
            q_values = model(stacked_state)
            action = q_values.argmax().item()
        if manual_start:
            action = 1
            manual_start = False
        next_state, reward, terminated, truncated, info = env.step(action)
        if info["lives"] < lives:
            manual_start = True
            lives = info["lives"]
        done = terminated or truncated

        stacked_state = stack_frames(next_state, frame_stack, scale=scale)
        stacked_state = np.expand_dims(stacked_state, axis=0)
        stacked_state = torch.FloatTensor(stacked_state).to(device)
        
        env.render()

if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    path = "FINAL10.pth"
    play_game(env, path, scale=0.5)
    env.close()

