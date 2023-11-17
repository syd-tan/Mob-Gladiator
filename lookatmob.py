import MalmoPython
from collections import namedtuple
import json 
import math
from past.utils import old_div
import random
import time

EntityInfo = namedtuple('EntityInfo', 'x, y, z, name')
mob_list = [("Skeleton", "", 2.0),
                ("Spider", "", 0.9),
                ("Creeper", "", 1.7),
                ("Zombie", "", 1.9),
                ("Silverfish", "", 0.2)]
       

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
    delta_yaw = angvel(target_yaw, yaw, 50.0)
    delta_pitch = angvel(target_pitch, pitch, 50.0)
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

def lookAtMob(json, agent_host): #world state observation data
    current_x = json.get(u'XPos', 0)
    current_z = json.get(u'ZPos', 0)
    current_y = json.get(u'YPos', 0)
    if "entities" in json:
        entities = [EntityInfo(k["x"], k["y"], k["z"], k["name"]) for k in json["entities"]]
        target_ent = entities[-1]
        print(entities)
        print(target_ent)
        # Look up height of entity from our table:
        target = [e for e in mob_list if e[0] == target_ent.name]
        if target:
            target_height = target[0][2]
            # Calculate where to look in order to see it:
            target_yaw, target_pitch = calcYawAndPitchToMob(target_ent, current_x, current_y, current_z, target_height)
            # And point ourselves there:
            pointTo(agent_host, json, target_pitch, target_yaw, 0.5)