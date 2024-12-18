import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from extracting_features import *
import matplotlib.pyplot as plt
import ale_py

class DuelingDQN(nn.Module):
    def __init__(self, action_space_size, input_height, input_width):
        super(DuelingDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            # first layer
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            #2nd layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # third layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            
        )
        conv_output_height= (input_height - 8)//4 + 1
        conv_output_width=(input_width - 8)//4 + 1
        conv_output_height=(conv_output_height - 4)//2 + 1
        conv_output_width=(conv_output_width - 4)//2 + 1
        conv_output_height=(conv_output_height - 3)//1 + 1
        conv_output_width=(conv_output_width - 3)// 1 + 1
        conv_output_size = 64 * conv_output_height * conv_output_width        
        # fully conected later
        self.fc_layer = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # state value
        self.value_stream = nn.Sequential(
            nn.Linear(512, 1)
        )
        # acton advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, action_space_size)
        )
    def forward(self, x):
        x= self.conv_layers(x)  
        x= x.view(x.size(0),-1)  
        x= self.fc_layer(x)
        value= self.value_stream(x)
        advantage= self.advantage_stream(x)
        q_values= value + (advantage -advantage.mean())
        return q_values
    
    
# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory= deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

def shape_reward(reward, observation, last_observation, action,info,last_action):
    # process the frames for detection
    frame = process_frame(observation)
    last_frame = process_frame(last_observation)

    # mario position
    mario_position = detect_mario(frame)
    mario_x, mario_y = mario_position
    last_mario_position = detect_mario(last_frame)
    last_mario_x, last_mario_y = mario_position
    
    # other positions 
    barrels = detect_barrels(frame)
    global daisy_position, ladders, highest_position
    if daisy_position is None:
        daisy_x, daisy_y = detect_daisy(frame)
    if ladders is None:
        ladders = detect_ladders(frame)

    shaped_reward = 0

    if info["episode_frame_number"] < 100 and action == 1:
        shaped_reward += .5

    #reward time spent alive
    shaped_reward += min(info["episode_frame_number"] *.00002,1)

    # Reward for being higher
    shaped_reward += min((115 - mario_y) * .05, 1)

    if mario_y < last_mario_y:
        shaped_reward += 1

    # penatly for stalling
    threshold = 200
    if info["episode_frame_number"] % threshold == 0:
        if mario_y >= highest_position:
            shaped_reward -= 10.0

    # Reward for new peak height
    if mario_y < highest_position: 
        shaped_reward += 5
        highest_position = mario_y

    # Reward for climbing ladders
    if any(abs(mario_x - ladder_x) < 5 and abs(mario_y - ladder_y) < 5 for ladder_x, ladder_y in ladders):
        if action == 2:
            shaped_reward += 2
        if action == 2 and mario_y < last_mario_y:
            shape_reward += 3
    else:
        if action == 2:
            shaped_reward -= 1
        if action == 5:
            shaped_reward -= 1  
    
    # Reward for proximity to ladders
    for ladder_x,ladder_y in ladders:
        if abs(mario_y - ladder_y) < 5:
            dist = abs(mario_x - ladder_x)
            if dist > 0: shaped_reward += .5 / dist
            if dist < 5: shaped_reward += .6

    # penalize inactivty 
    if mario_position == last_mario_position: shaped_reward-= 2

    for barrel_x, barrel_y in barrels:
        if abs(mario_x - barrel_x) < 5:
            #penatly for touching barrel
            if abs(mario_y - barrel_y) < 5: 
                shaped_reward -= 4
            #reward for jumping over
            elif mario_y - barrel_y < -5: 
                shaped_reward += 4
        #small dange for being near a barrel
        if abs(mario_x - barrel_x) < 10 and abs(mario_y - barrel_y) < 10:
            shaped_reward -= .01

    # scaling down in game rewards
    shaped_reward += reward/20

    #reward for distance to daisy
    distance_to_daisy = np.sqrt((mario_x - daisy_x)**2 + (mario_y - daisy_y)**2)
    if distance_to_daisy > 0: shaped_reward += min(.1 / distance_to_daisy+ .0000001, .5)

    return shaped_reward

def train_dqn_with_scaling(env, episodes=100, batch_size=32, learning_rate=0.001, gamma=0.98, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99, replay_capacity=100000, scale=0.5):
    height = int(210*scale)
    width = int(160*scale)
    action_space = env.action_space.n
    policy_net = DuelingDQN(action_space, height, width).to(device)
    target_net = DuelingDQN(action_space, height, width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = F.smooth_l1_loss
    replay_memory = ReplayMemory(replay_capacity)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    epsilon = epsilon_start
    rewards_per_epsiode = []

    for episode in range(episodes):
        state = env.reset()[0]
        frame_stack = deque(maxlen=4)
        stacked_state = stack_frames(state, frame_stack, scale=scale)
        stacked_state = np.expand_dims(stacked_state, axis=0)
        stacked_state = torch.FloatTensor(stacked_state).to(device)

        last_observation =state
        total_reward = 0
        done = False
        last_action = None

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = policy_net(stacked_state).argmax().item()
        
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            shaped_reward = shape_reward(reward, next_state, last_observation, action,info,last_action)
            stacked_next_state = stack_frames(next_state, frame_stack, scale=scale)
            stacked_next_state = np.expand_dims(stacked_next_state, axis=0)
            stacked_next_state = torch.FloatTensor(stacked_next_state).to(device)
            replay_memory.append((stacked_state, action, shaped_reward, stacked_next_state, done))
            stacked_state = stacked_next_state
            last_observation = next_state
            total_reward += shaped_reward
            last_action = action

        if len(replay_memory) > batch_size:
            transitions = replay_memory.sample(batch_size)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
            batch_state = torch.cat(batch_state)
            batch_action = torch.LongTensor(batch_action).to(device)
            batch_reward = torch.FloatTensor(batch_reward).to(device)
            batch_next_state = torch.cat(batch_next_state)
            batch_done = torch.BoolTensor(batch_done).to(device)
            current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = target_net(batch_next_state).max(1)[0]
                target_q_values = batch_reward + gamma * max_next_q_values * (~batch_done)
            loss = loss_fn(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
        epsilon = max(epsilon * epsilon_decay,epsilon_end)

        if episode %10==0:
            target_net.load_state_dict(policy_net.state_dict())

        scheduler.step()
        rewards_per_epsiode.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f}")

    path = "FINAL10.pth"
    torch.save(policy_net.state_dict(), path)
    print(f"Model saved to {path}")
    return rewards_per_epsiode

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    daisy_position= None
    ladders = None
    highest_position = float("inf")
    gym.register_envs(ale_py)
    # env = gym.make("ALE/DonkeyKong-v5",render_mode="human")
    env = gym.make("ALE/DonkeyKong-v5")
    rewards = train_dqn_with_scaling(env, episodes=200)
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()
