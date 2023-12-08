from __future__ import print_function
from __future__ import division

# Fighting Mobs

from builtins import range
from datetime import datetime, timedelta
from past.utils import old_div
import json
import math

from state import agent_state

# Task parameters:
ARENA_WIDTH = 15
ARENA_BREADTH = 15

# Reward Constants
REWARD_PER_DAMAGE_DEALT = 8 
REWARD_PER_DAMAGE_TAKEN = -2
REWARD_PER_ATTACK = -4
REWARD_FOR_STAYING_ALIVE_PER_TICK = -0.25
REWARD_ENEMY_DEAD = 2000
REWARD_PLAYER_DEATH = -2000
REWARD_OUT_OF_TIME = -3000


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

def getMissionXML(summary, msPerTick, video_requirements):
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
               <ServerQuitFromTimeUp timeLimitMs="30000"description="out_of_time"/>
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
                {video_requirements}
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
            pos_x, pos_y, pos_z, vel_x, vel_z = entity["x"], entity["y"], entity["z"], entity["motionX"], entity["motionZ"]
            return agent_health, vertical_motion, pos_x, pos_y, pos_z, vel_x, vel_z
    return 0, 0, 0, 0, 0, 0


def step(agent_host, world_state, curr_state, enemy_mob, finish_time):    
    obs_json = json.loads(world_state.observations[0].text)
    done = False
    agent_health, vertical_motion, pos_x, pos_y, pos_z, vel_x, vel_z = get_agent_states(obs_json)

    time_diff = int((curr_state.get_cooldown_completion_time() - datetime.now()).total_seconds()) if curr_state else 0
    attack_cooldown_remaining = time_diff if time_diff > 0 else 0

    mob_health, opp_pos_x, opp_pos_y, opp_pos_z = get_opponent_states(obs_json, enemy_mob)
    in_range = obs_json["LineOfSight"]["inRange"] if "LineOfSight" in obs_json else False

    distance = math.sqrt(
        math.pow(pos_x - opp_pos_x, 2)
        + math.pow(pos_y - opp_pos_y, 2)
        + math.pow(pos_z - opp_pos_z, 2)
    ) if mob_health != 0 and agent_health != 0 else -1

    mission_time_remaining = int((finish_time - datetime.now()).total_seconds())
    next_state = agent_state(
        enemy=enemy_mob,
        enemy_health=mob_health,
        agent_health=agent_health,
        agent_x=pos_x,
        agent_z=pos_z,
        velocity_x=vel_x,
        velocity_z=vel_z,
        vertical_motion=vertical_motion,
        distance_from_enemy=distance,
        attack_cooldown_remaining=attack_cooldown_remaining,
        in_range=in_range,
        time_remaining=mission_time_remaining,
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