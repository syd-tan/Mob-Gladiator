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
            

def get_mob():
    mobs = ["Skeleton", "Creeper", "Zombie", "Spider"]
    chosen_mob = mobs[random.randint(0, len(mobs) - 1)]
    return chosen_mob

def summon_mob(agent_host, chosen_mob):
    x_coord = random.choice([random.uniform(-6.25, -5.5), random.uniform(5.5, 6.25)])
    z_coord = random.choice([random.uniform(-6.25, -5.5), random.uniform(5.5, 6.25)])
    if chosen_mob == "Zombie":
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord} {{IsBaby:0}}")
    elif chosen_mob == "Spider":
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord} {{Passengers:[]}}")
    elif chosen_mob == "Skeleton":
        agent_host.sendCommand(f'chat /summon {chosen_mob} {x_coord} 207 {z_coord} {{HandItems:[{{id:"minecraft:bow",Count:1b}}],ArmorItems:[{{}},{{}},{{}},{{}}]}}')
    else:
        agent_host.sendCommand(f"chat /summon {chosen_mob} {x_coord} 207 {z_coord}")

def initialize_mission(agent_host, chosen_mob):
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    agent_host.sendCommand("chat /gamerule naturalRegeneration false")
    summon_mob(agent_host, chosen_mob)
    world_state = wait_until_opponent_spawns(world_state, agent_host)
    return world_state