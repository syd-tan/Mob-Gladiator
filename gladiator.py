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
import MalmoPython
import random
import time
import json
import random
import math
import malmoutils

from state import agent_state
from lookatmob import lookAtMob

malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)

# Task parameters:
ARENA_WIDTH = 20
ARENA_BREADTH = 20

# Reward Constants
REWARD_PER_DAMAGE_DEALT = 10
REWARD_ENEMY_DEAD = 1000
REWARD_PLAYER_DEATH = -3000
REWARD_OUT_OF_TIME = -1000


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
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="{str(ARENA_WIDTH + 5)}" yrange="10" zrange="{str(ARENA_BREADTH + 5)}" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
            </AgentHandlers>
        </AgentSection>

    </Mission>"""


def wait_until_opponent_spawns(world_state):
    while True:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if world_state.observations:
            observations = json.loads(world_state.observations[0].text)
            if len(observations["entities"]) > 1:
                return world_state

def summon_mob(agent_host):
    mobs = ["Skeleton", "Creeper", "Zombie", "Spider"]
    chosen_mob = mobs[random.randint(0, len(mobs) - 1)]
    x_coord = random.choice([random.uniform(-9.5, -6), random.uniform(6, 9.5)])
    z_coord = random.choice([random.uniform(-9.5, -6), random.uniform(6, 9.5)])
    if chosen_mob == "Zombie":
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord} {{IsBaby:0}}")
    elif chosen_mob == "Spider":
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord} {{Passengers:[]}}")
    else:
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord}")

   
    return chosen_mob


def initialize_mission(agent_host):
    agent_host.sendCommand("chat /gamerule naturalRegeneration false")
    return summon_mob(agent_host)

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


def step(agent_host, world_state, curr_state, enemy_mob, next_attack_time):
    observations_json = json.loads(world_state.observations[0].text)
    done = False
    is_enemy_alive = any(entity["name"] == enemy_mob for entity in observations_json["entities"])
    agent_health, vertical_motion, pos_x, pos_y, pos_z = get_agent_states(
            observations_json
        )
    reward = sum (world_state.rewards[i].getValue() for i in range(len(world_state.rewards)))
    if is_enemy_alive:
        mob_health, opp_pos_x, opp_pos_y, opp_pos_z = get_opponent_states(
            observations_json, enemy_mob
        )
        in_range = observations_json["LineOfSight"]["inRange"]
        distance = math.sqrt(
            math.pow(pos_x - opp_pos_x, 2)
            + math.pow(pos_y - opp_pos_y, 2)
            + math.pow(pos_z - opp_pos_z, 2)
        )

        attack_cooldown = 0
        if next_attack_time > datetime.now():
            attack_cooldown = next_attack_time - datetime.now()

        gladiator_state = agent_state(
            enemy=enemy_mob,
            enemy_health=mob_health,
            agent_health=agent_health,
            vertical_motion=vertical_motion,
            distance_from_enemy=distance,
            attack_cooldown=attack_cooldown,
            in_range=in_range,
        )

        # Updating rewards 
        if curr_state:
            damage_dealt = curr_state.get_enemy_health() - mob_health
            reward += (damage_dealt * REWARD_PER_DAMAGE_DEALT)

        return gladiator_state, reward, done
    else:
        # TODO: Set state after mob dies 
        # The mob is no longer an observable entity so this is needed
        agent_host.sendCommand("quit")
        done = True
        return None, reward, done
    


def act(agent_host, action, next_attack_time):
    agent_host.sendCommand(action)
    if action == "attack 1":
        agent_host.sendCommand("attack 0")
        next_attack_time = datetime.now() + timedelta(seconds=0.625)
    elif action == "jump 1":
        agent_host.sendCommand("jump 0")

    # TODO: UPDATE attack cooldown for next state

    


# Add Minecraft Client
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

msPerTick = 50  # 50 ms per tick is default
next_attack_time = datetime.now()

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000

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
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    # set gamerule and spawn
    enemy_mob = initialize_mission(agent_host)

    # waits until the opponent fully spawn in order to be detected by malmo
    world_state = wait_until_opponent_spawns(world_state)

    # initialize mission settings
    total_reward = 0
    curr_state = None
    curr_state, _, _ = step(agent_host, world_state, curr_state, enemy_mob, next_attack_time)
    # main loop
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        reward = 0
        if world_state.observations:
            # action, next_attack_time = act(curr_state, next_attack_time)
            next_state, reward, done = step(agent_host, world_state, curr_state, enemy_mob, next_attack_time)
            print(reward)
            lookAtMob(world_state, agent_host, enemy_mob)
            agent_host.sendCommand("attack 1")
            agent_host.sendCommand("move 1")

            # remember
            # train

            curr_state = next_state

        total_reward += reward            

    # mission has ended.
    for error in world_state.errors:
        print("Error:", error.text)

    print()
    print("=" * 41)
    print("Total score this round:", total_reward)
    print("=" * 41)
    print()
    time.sleep(1)  # Give the mod a little time to prepare for the next mission.
