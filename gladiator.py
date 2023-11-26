from __future__ import print_function
from __future__ import division

# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Demo of reward for damaging mobs - create an arena filled with pigs and sheep,
# and reward the agent positively for attacking sheep, and negatively for attacking pigs.
# Using this reward signal to train the agent is left as an exercise for the reader...
# this demo just uses ObservationFromRay and ObservationFromNearbyEntities to determine
# when and where to attack.

from builtins import range
from datetime import datetime, timedelta
from past.utils import old_div
import numpy as np
import MalmoPython
import random
import time
import json
import random
import math
import malmoutils

from state import agent_state
from initialize_mission import initialize_mission
from lookatmob import lookAtMob

malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)

# Task parameters:
ARENA_WIDTH = 20
ARENA_BREADTH = 20

# Reward Constants
REWARD_PER_DAMAGE_DEALT = 2
REWARD_PER_DAMAGE_TAKEN = -1
REWARD_PER_ATTACK = -5
REWARD_FOR_STAYING_ALIVE_PER_TICK = 0.25
REWARD_ENEMY_DEAD = 3000
REWARD_PLAYER_DEATH = -3000
REWARD_OUT_OF_TIME = -2000


def getCorner(index, top, left, expand=0, y=0):
    """Return part of the XML string that defines the requested corner"""
    x = (
        str(-(expand + old_div(ARENA_WIDTH, 2)))
        if left
        else str(expand + old_div(ARENA_WIDTH, 2))
    )
    z = (
        str(-(expand + old_div(ARENA_BREADTH, 2)))
        if top
        else str(expand + old_div(ARENA_BREADTH, 2))
    )
    return f'x{index}="{x}" y{index}="{y}" z{index}="{z}"'


def getMissionXML(summary, msPerTick):
    """Build an XML mission string."""
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>{summary}</Summary>
        </About>

        <ModSettings>
            <MsPerTick>{msPerTick}</MsPerTick>
        </ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>18000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <AllowSpawning>false</AllowSpawning>
                <AllowedMobs>Zombie Skeleton Spider Creeper</AllowedMobs>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;1*minecraft:bedrock,7*minecraft:dirt,1*minecraft:grass;1;" />
                <DrawingDecorator>
                    <DrawCuboid {getCorner("1", True, True, expand=10, y=206)} {getCorner("2", False, False, y=215, expand=10)} type="stone"/>
                    <DrawCuboid {getCorner("1", True, True, y=207)} {getCorner("2", False, False, y=215)} type="glass"/>
                    <DrawCuboid {getCorner("1", True, True, y=207)} {getCorner("2", False, False, y=214)} type="air"/>
                </DrawingDecorator>
               <ServerQuitWhenAnyAgentFinishes />
               <ServerQuitFromTimeUp timeLimitMs="60000"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Agent</Name>
            <AgentStart>
                <Placement x="0.5" y="207.0" z="0.5" pitch="20"/>
                <Inventory>
                    <InventoryItem type="wooden_sword" slot="0"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="420"/>
                <ChatCommands />
                <MissionQuitCommands quitDescription="enemy_dead"/>
                <ObservationFromRay/>
                <RewardForMissionEnd rewardForDeath="{REWARD_PLAYER_DEATH}">
                    <Reward description="out_of_time" reward="{REWARD_OUT_OF_TIME}"/>
                    <Reward description="enemy_dead" reward="{REWARD_ENEMY_DEAD}"/>
                </RewardForMissionEnd>
                <RewardForTimeTaken initialReward="0" delta="{REWARD_FOR_STAYING_ALIVE_PER_TICK}" density="PER_TICK" />
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="{str(ARENA_WIDTH + 5)}" yrange="10" zrange="{str(ARENA_BREADTH + 5)}" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
            </AgentHandlers>
        </AgentSection>

    </Mission>"""


def check_mission_over(agent_host, world_state):
    observations_json = json.loads(world_state.observations[0].text)
    if len(observations_json["entities"]) == 1:
        agent_host.sendCommand("quit")
        return True
    return False


def get_opponent_states(observations, entity_name):
    for entity in observations["entities"]:
        if entity["name"] == entity_name:
            opponent_health = entity["life"]
            pos_x, pos_y, pos_z = entity["x"], entity["y"], entity["z"]
            return opponent_health, pos_x, pos_y, pos_z
    return 0, 0, 0, 0


def get_agent_states(observations):
    for entity in observations["entities"]:
        if entity["name"] == "Agent":
            agent_health = entity["life"]
            vertical_motion = (
                entity["motionY"] / abs(entity["motionY"])
                if entity["motionY"] != 0
                else 0
            )
            pos_x, pos_y, pos_z = entity["x"], entity["y"], entity["z"]
            return agent_health, vertical_motion, pos_x, pos_y, pos_z
    return 0, 0, 0, 0


def step(agent_host, world_state, curr_state, enemy_mob):    
    obs_json = json.loads(world_state.observations[0].text)
    done = False
    agent_health, vertical_motion, pos_x, pos_y, pos_z = get_agent_states(obs_json)

    time_diff = int((curr_state.get_cooldown_completion_time() - datetime.now()).total_seconds()) if curr_state else 0
    attack_cooldown_remaining = time_diff if time_diff > 0 else 0

    mob_health, opp_pos_x, opp_pos_y, opp_pos_z = get_opponent_states(obs_json, enemy_mob)
    in_range = obs_json["LineOfSight"]["inRange"] if "LineOfSight" in obs_json else False

    distance = math.sqrt(
        math.pow(pos_x - opp_pos_x, 2)
        + math.pow(pos_y - opp_pos_y, 2)
        + math.pow(pos_z - opp_pos_z, 2)
    ) if mob_health != 0 and agent_health != 0 else 0

    next_state = agent_state(
        enemy=enemy_mob,
        enemy_health=mob_health,
        agent_health=agent_health,
        vertical_motion=vertical_motion,
        distance_from_enemy=distance,
        attack_cooldown_remaining=attack_cooldown_remaining,
        in_range=in_range,
    )

    # Quit if mob is dead
    if mob_health == 0 and agent_health != 0:
        agent_host.sendCommand("quit")

    # Updating rewards
    reward = sum(world_state.rewards[i].getValue() for i in range(len(world_state.rewards)))
    if curr_state:
        # Damage Dealt
        damage_dealt = obs_json["DamageDealt"] - curr_state.get_damage_dealt()
        damage_dealt_reward = damage_dealt * REWARD_PER_DAMAGE_DEALT
        # Damage Taken
        damage_taken = obs_json["DamageTaken"] - curr_state.get_damage_taken()
        damage_taken_reward = damage_taken * REWARD_PER_DAMAGE_TAKEN
        # Attack Spamming
        attacking_negative_reward = REWARD_PER_ATTACK if curr_state.get_action_performed() == "attack 1" else 0

        reward += damage_dealt_reward + damage_taken_reward + attacking_negative_reward
    next_state.set_damage_dealt(obs_json["DamageDealt"])
    next_state.set_damage_taken(obs_json["DamageTaken"])

    return next_state, reward, done


def perform_action(agent_host, action, curr_state):
    agent_host.sendCommand(action)
    curr_state.set_action_performed(action)
    if action == "attack 1":
        agent_host.sendCommand("attack 0")
        curr_state.set_cooldown_completion_time(datetime.now() + timedelta(seconds=0.625))
    elif action == "jump 1":
        agent_host.sendCommand("jump 0")


# Add Minecraft Client
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

msPerTick = 50  # 50 ms per tick is default

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000
# Evaluate: Initialize episode rewards array
episode_rewards = [0 for _ in range(num_reps)]
rewards_per_monster = {}
attacks_per_monster = {}
remaining_agent_hp_per_monster = {}
winrates_per_monster = {}

for iRepeat in range(num_reps):
    mission_xml = getMissionXML("Gladiator Begin! #" + str(iRepeat), msPerTick)
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
    enemy_mob, world_state = initialize_mission(agent_host, world_state)
    # adds mob to rewards+attacks per enemy if not initialized
    if enemy_mob not in rewards_per_monster:
        rewards_per_monster[enemy_mob] = []
        attacks_per_monster[enemy_mob] = []
        remaining_agent_hp_per_monster[enemy_mob] = []
        winrates_per_monster[enemy_mob] = (0, 0)

    # initialize mission settings
    total_reward = 0
    total_attacks = 0
    curr_state = None
    curr_state, _, _ = step(agent_host, world_state, curr_state, enemy_mob)
    # main loop
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()

        reward = 0
        # retrieve updates in rewards per tick and at missions ends
        if world_state.rewards:
            reward += world_state.rewards[-1].getValue()
        
        # when the world state has observations, the action and processes are rerun
        if world_state.observations:
            action = (
                "attack 1" if random.choice([True, False]) else "move 1"
            )  # TODO: act() should be run here to get the action
            if "attack" in action: total_attacks += 1
            perform_action(agent_host, action, curr_state)
            next_state, step_reward, done = step(agent_host,world_state,curr_state,enemy_mob)
            reward += step_reward
            lookAtMob(world_state, agent_host, enemy_mob)

            # remember
            # train

            curr_state = next_state
        total_reward += reward

    # mission has ended.
    for error in world_state.errors:
        print("Error:", error.text)
    # Evaluate: Change episode reward in array and add reward/attacks to records for that mob
    episode_rewards[iRepeat] = total_reward
    rewards_per_monster[enemy_mob].append(total_reward)
    attacks_per_monster[enemy_mob].append(total_attacks)
    remaining_agent_hp_per_monster[enemy_mob].append(curr_state.agent_health if curr_state.agent_health > 0 else 0)
    winrates_per_monster[enemy_mob] = (
        winrates_per_monster[enemy_mob][0] + (1 if curr_state.agent_health > 0 else 0),
        winrates_per_monster[enemy_mob][1] + 1
    )

    print()
    print("=" * 41)
    print("Total score this round:", total_reward)
    print("=" * 41)
    print()
    time.sleep(1)  # Give the mod a little time to prepare for the next mission.

# Evaluate: Summary statistics of rewards
# print(attacks_per_monster)
# print(remaining_agent_hp_per_monster)
# print(winrates_per_monster)
print(f'Mean reward of {num_reps} episodes:', np.mean(episode_rewards))
print(f'Std deviation of {num_reps} episodes:', np.std(episode_rewards))
print('Per monster:')
max_length_mobname = len(max(rewards_per_monster.keys(), key=len))
for mob, r in sorted(rewards_per_monster.items(), key=lambda k: k[0]):
    print(f'{mob.ljust(max_length_mobname + 1)}| {np.mean(r)} (std. dev {np.std(r)}, {round(winrates_per_monster[mob][0]/winrates_per_monster[mob][1] * 100, 2)}% winrate)')