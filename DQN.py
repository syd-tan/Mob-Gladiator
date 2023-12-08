# reference : https://jolly-balaur-c5d.notion.site/Code-DQN-da35f34a0a0c48d4be5fc5b1411bdac7

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from pathlib import Path


device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99999
LEARNING_RATE = 0.0005
TARGET_UPDATE_FREQ = 10


SAVE_CHECKPOINT_FREQ = 100

class DQN:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = []
        self.epsilon = EPSILON_START
        # Evaluate: Initialize episode stats
        # episode_rewards = []
        self.rewards_per_monster = {}
        self.attacks_per_monster = {}
        self.remaining_agent_hp_per_monster = {}
        self.winrates_per_monster = {}

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > BUFFER_SIZE:
            self.memory.pop(0)
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        curr_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_q_values * (~dones))

        loss = nn.MSELoss()(curr_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, episode):  
        torch.save({
            'last_saved_ep': episode,
            'optimizer': self.optimizer.state_dict(),
            'target_network': self.target_network.state_dict(),
            'memory': self.memory,
            'epsilon': self.epsilon,
            'q_network':self.q_network.state_dict()
        }, f'training_model.tar')
        # Evaluate: Save checkpoint stats to json
        script_dir = Path(__file__).parent
        with script_dir.joinpath('stats', f'stats-checkpoint-{episode}.json').open(mode='w') as eval_data:
            json.dump(
                {
                    "rep": episode,
                    "rewards_per_monster": self.rewards_per_monster,
                    "attacks_per_monster": self.attacks_per_monster,
                    "remaining_agent_hp_per_monster": self.remaining_agent_hp_per_monster,
                    "winrates_per_monster": self.winrates_per_monster
                },
                eval_data,
                indent=2
            )
        # Evaluate: Reset rewards structures to free up memory
        for mob in self.rewards_per_monster.keys():
            self.rewards_per_monster[mob] = []
            self.attacks_per_monster[mob] = []
            self.remaining_agent_hp_per_monster[mob] = []
            self.winrates_per_monster[mob] = (0, 0)
        
    def load_checkpoint(self, checkpoint):
        if type(checkpoint['epsilon']) == float:
            self.epsilon = checkpoint['epsilon']
        else:
            self.epsilon = checkpoint['epsilon'][0]
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.memory = checkpoint['memory']
        self.q_network.load_state_dict(checkpoint['q_network'])