from collections import namedtuple
import json 
import math
from past.utils import old_div

EntityInfo = namedtuple('EntityInfo', 'x, y, z, name')
mob_list = [("Skeleton", "", 2.0),
                ("Spider", "", 0.9),
                ("Creeper", "", 1.7),
                ("Zombie", "", 1.9),
                ("Silverfish", "", 0.2)]
mob_set = {"Skeleton":1.7, "Spider":.6, "Creeper":1.5, "Zombie":1.5}
       

def angvel(target, current, scale):
    '''Use sigmoid function to choose a delta that will help smoothly steer from current angle to target angle.'''
    delta = target - current
    while delta < -180:
        delta += 360
    while delta > 180:
        delta -= 360
    return (old_div(2.0, (1.0 + math.exp(old_div(-delta,scale))))) - 1.0

def pointTo(agent_host, ob, target_pitch, target_yaw, threshold):
    '''Steer towards the target pitch/yaw, return True when within the given tolerance threshold.'''
    pitch = ob.get(u'Pitch', 0)
    yaw = ob.get(u'Yaw', 0)
    delta_yaw = angvel(target_yaw, yaw, 40.0)
    delta_pitch = angvel(target_pitch, pitch, 40.0)
    agent_host.sendCommand("turn " + str(delta_yaw))    
    agent_host.sendCommand("pitch " + str(delta_pitch))
    if abs(pitch-target_pitch) + abs(yaw-target_yaw) < threshold:
        agent_host.sendCommand("turn 0")
        agent_host.sendCommand("pitch 0")
        return True
    return False

def calcYawAndPitchToMob(target, x, y, z, target_height):
    dx = target.x - x
    dz = target.z - z
    yaw = -180 * math.atan2(dx, dz) / math.pi
    distance = math.sqrt(dx * dx + dz * dz)
    pitch = math.atan2(((y + 1.625) - (target.y + target_height * 0.9)), distance) * 180.0 / math.pi
    return yaw, pitch

def lookAtMob(world_state, agent_host, name): #world state observation data
    observation_json = json.loads(world_state.observations[-1].text)
    current_x = observation_json.get(u'XPos', 0)
    current_z = observation_json.get(u'ZPos', 0)
    current_y = observation_json.get(u'YPos', 0)
    if "entities" in observation_json:
        entities = [EntityInfo(k["x"], k["y"], k["z"], k["name"]) for k in observation_json["entities"]]
        targets = [e for e in entities if e.name in mob_set]
        # Look up height of entity from our table:
        if targets:
            target_ent = targets[0]
            target_height = mob_set[name]
            # Calculate where to look in order to see it:
            target_yaw, target_pitch = calcYawAndPitchToMob(target_ent, current_x, current_y, current_z, target_height)
            # And point ourselves there:
            pointTo(agent_host, observation_json, target_pitch, target_yaw, 0.5)