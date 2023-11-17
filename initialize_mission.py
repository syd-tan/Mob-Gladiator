import random
import json
import time

def wait_until_opponent_spawns(world_state, agent_host):
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


def initialize_mission(agent_host, world_state):
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    agent_host.sendCommand("chat /gamerule naturalRegeneration false")
    enemy_mob = summon_mob(agent_host)
    world_state = wait_until_opponent_spawns(world_state, agent_host)
    return enemy_mob, world_state