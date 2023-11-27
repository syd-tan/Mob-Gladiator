# reference : https://jolly-balaur-c5d.notion.site/Code-DQN-da35f34a0a0c48d4be5fc5b1411bdac7

from pathlib import Path
import json
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
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
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
        }, 'training_model.tar')
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


if __name__ == "__main__":
    actions_list = [
        "move 1",
        "move -1",
        "move 0",
        "strafe 1",
        "strafe -1",
        "strafe 0",
        "jump 1",
        "attack 1",
    ]
    agent = DQN(8, 8)
    agent_host = MalmoPython.AgentHost()
    # Add Minecraft Client
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    msPerTick = 50  # 50 ms per tick is default

    if agent_host.receivedArgument("test"):
        num_reps = 1
    else:
        num_reps = 30000

    try: 
        checkpoint = torch.load('training_model.tar')
        
        start = checkpoint['last_saved_ep'] + 1
        agent.load_checkpoint(checkpoint)
    except (FileNotFoundError): 
        start = 1

    for iRepeat in range(start, num_reps + 1):
        print(iRepeat)
        mission_xml = gladiator.getMissionXML(
            "Gladiator Begin! #" + str(iRepeat), msPerTick
        )
        
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
        # adds mob to rewards+attacks per enemy if not initialized
        if enemy_mob not in agent.rewards_per_monster:
            agent.rewards_per_monster[enemy_mob] = []
            agent.attacks_per_monster[enemy_mob] = []
            agent.remaining_agent_hp_per_monster[enemy_mob] = []
            agent.winrates_per_monster[enemy_mob] = (0, 0)

        # initialize mission settings
        total_reward = 0
        total_attacks = 0
        curr_state = None
        curr_state, _, _ = gladiator.step(
            agent_host, world_state, curr_state, enemy_mob
        )
        # main loop
        while world_state.is_mission_running:
            world_state = agent_host.getWorldState()

            reward = 0
            # retrieve updates in rewards per tick and at missions ends
            if world_state.rewards:
                reward += world_state.rewards[-1].getValue()

            # when the world state has observations, the action and processes are rerun
            if world_state.observations:
                # action
                curr_state_dqn = curr_state.get_state()
                action_index = agent.act(curr_state_dqn)
                action = actions_list[action_index]
                if "attack" in action: total_attacks += 1

                # step
                gladiator.perform_action(agent_host, action, curr_state)
                next_state, step_reward, done = gladiator.step(agent_host, world_state, curr_state, enemy_mob)
                reward += step_reward
                lookatmob.lookAtMob(world_state, agent_host, enemy_mob)

                # remember
                next_state_dqn = next_state.get_state()
                agent.remember(curr_state_dqn, action_index, reward, next_state_dqn, done)
                # train
                agent.train()
                
                # Update state
                curr_state = next_state
            total_reward += reward

        if iRepeat % SAVE_CHECKPOINT_FREQ == 0:
            agent.save_checkpoint(iRepeat)
        if iRepeat % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # mission has ended.
        for error in world_state.errors:
            print("Error:", error.text)
        # Evaluate: Change episode reward in array and add reward/attacks to records for that mob
        agent.episode_rewards[iRepeat] = total_reward
        agent.rewards_per_monster[enemy_mob].append(total_reward)
        agent.attacks_per_monster[enemy_mob].append(total_attacks)
        agent.remaining_agent_hp_per_monster[enemy_mob].append(curr_state.agent_health if curr_state.agent_health > 0 else 0)
        agent.winrates_per_monster[enemy_mob] = (
            agent.winrates_per_monster[enemy_mob][0] + (1 if curr_state.agent_health > 0 else 0),
            agent.winrates_per_monster[enemy_mob][1] + 1
        )

        print()
        print("=" * 41)
        print("Total score this round:", total_reward)
        print("=" * 41)
        print()
        time.sleep(1)  # Give the mod a little time to prepare for the next mission.