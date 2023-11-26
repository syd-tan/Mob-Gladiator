# reference : https://jolly-balaur-c5d.notion.site/Code-DQN-da35f34a0a0c48d4be5fc5b1411bdac7

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import state
import gladiator
import MalmoPython
import time
import lookatmob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0005
TARGET_UPDATE_FREQ = 10

class DQN:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.memory = []
        self.epsilon = EPSILON_START

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

if __name__ == "__main__":
    agent = DQN(7, 8)
    agent_host = MalmoPython.AgentHost()
    # Add Minecraft Client
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    msPerTick = 50  # 50 ms per tick is default

    if agent_host.receivedArgument("test"):
        num_reps = 1
    else:
        num_reps = 30000

    for iRepeat in range(num_reps):
        mission_xml = gladiator.getMissionXML("Gladiator Begin! #" + str(iRepeat), msPerTick)
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Attempt to start the mission:
                agent_host.startMission(
                    my_mission, my_client_pool, my_mission_record, 0, "MobGladiator"
                )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission", e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()

        # initializes mission
        enemy_mob, world_state = gladiator.initialize_mission(agent_host, world_state)

        # initialize mission settings
        total_reward = 0
        curr_state = None
        curr_state, _, _ = gladiator.step(agent_host, world_state, curr_state, enemy_mob)
        # main loop
        while world_state.is_mission_running:
            world_state = agent_host.getWorldState()

            reward = 0
            # retrieve updates in rewards per tick and at missions ends
            if world_state.rewards:
                reward += world_state.rewards[-1].getValue()
            
            # when the world state has observations, the action and processes are rerun
            if world_state.observations:
                # action = (
                #     "attack 1" if random.choice([True, False]) else "move 1"
                # )  
                action = agent.act(curr_state) # TODO: act() should be run here to get the action
                gladiator.perform_action(agent_host, action, curr_state)
                next_state, step_reward, done = gladiator.step(agent_host,world_state,curr_state,enemy_mob)
                reward += step_reward
                lookatmob.lookAtMob(world_state, agent_host, enemy_mob)

                # remember
                agent.remember(curr_state, action, reward, next_state, done)
                # train
                agent.train()

                curr_state = next_state
            total_reward += reward

        if iRepeat % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            
        # mission has ended.
        for error in world_state.errors:
            print("Error:", error.text)

        print()
        print("=" * 41)
        print("Total score this round:", total_reward)
        print("=" * 41)
        print()
        time.sleep(1)  # Give the mod a little time to prepare for the next mission.