import MalmoPython
from collections import namedtuple
import json 
import math
from past.utils import old_div
import random
import time

ARENA_WIDTH = 25
ARENA_BREADTH = 25

agent_host = MalmoPython.AgentHost()
video_requirements = '<VideoProducer><Width>860</Width><Height>480</Height></VideoProducer>' if agent_host.receivedArgument("record_video") else ''


EntityInfo = namedtuple('EntityInfo', 'x, y, z, name')
mob_list = [("Skeleton", "", 2.0),
                ("Spider", "", 0.9),
                ("Creeper", "", 1.7),
                ("Zombie", "", 1.9)]

def getCorner(index,top,left,expand=0,y=0):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+old_div(ARENA_WIDTH,2))) if left else str(expand+old_div(ARENA_WIDTH,2))
    z = str(-(expand+old_div(ARENA_BREADTH,2))) if top else str(expand+old_div(ARENA_BREADTH,2))
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'\

def getMobSpawnDetails():
    mobs = ["Zombie", "Skeleton", "Creeper", "Spider"]
    chosen_mob = mobs[random.randint(0, len(mobs) - 1)]
    return f'x="-11" y="207" z="11" type="Skeleton"'

def getMissionXML(summary, msPerTick):
    ''' Build an XML mission string.'''
    # We put the spawners inside an animation object, to move them out of range of the player after a short period of time.
    # Otherwise they will just keep spawning - as soon as the agent kills a sheep, it will be replaced.
    # (Could use DrawEntity to create the pigs/sheep, rather than using spawners... but this way is much more fun.)
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>''' + str(msPerTick) + '''</MsPerTick>
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
                    <DrawCuboid ''' + getCorner("1",True,True,expand=10,y=206) + " " + getCorner("2",False,False,y=215,expand=10) + ''' type="stone"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=207) + " " + getCorner("2",False,False,y=215) + ''' type="glass"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=207) + " " + getCorner("2",False,False,y=214) + ''' type="air"/>
                </DrawingDecorator>
                <DrawingDecorator>
                    <DrawEntity ''' + getMobSpawnDetails() + ''' />
                </DrawingDecorator>
               <ServerQuitWhenAnyAgentFinishes />
               <ServerQuitFromTimeUp timeLimitMs="60000"/>
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>The Hunter</Name>
            <AgentStart>
                <Placement x="0.5" y="207.0" z="0.5" pitch="20"/>
                <Inventory>
                    <InventoryItem type="wooden_sword" slot="0"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="420"/>
                <ChatCommands />
                <ObservationFromRay/>
                <RewardForDamagingEntity>
                    <Mob type="Zombie" reward="1"/>
                    <Mob type="Skeleton" reward="1"/>
                    <Mob type="Creeper" reward="1"/>
                    <Mob type="Spider" reward="1"/>
                </RewardForDamagingEntity>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>''' + video_requirements + '''
            </AgentHandlers>
        </AgentSection>

    </Mission>'''

def checkEnts(present_entities, required_entities):
    missing = []
    for ent in required_entities:
        if not ent in present_entities:
            missing.append(ent)
    if len(missing) > 0:
        print("Can't find:", missing)
       

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


my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

world_state = agent_host.getWorldState()
my_mission = MalmoPython.MissionSpec(getMissionXML("", 50), True)
my_mission_record = MalmoPython.MissionRecordSpec()
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_client_pool, my_mission_record, 0, "blahblah" )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission",e)
            print("Is the game running?")
            exit(1)
        else:
            time.sleep(2)
while not world_state.has_mission_begun:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

while world_state.is_mission_running:
    world_state = agent_host.getWorldState()
    # 
    if world_state.number_of_observations_since_last_state > 0:
        obvsText = world_state.observations[-1].text
        data = json.loads(obvsText) # observation comes in as a JSON string...
        current_x = data.get(u'XPos', 0)
        current_z = data.get(u'ZPos', 0)
        current_y = data.get(u'YPos', 0)
        if "entities" in data:
            entities = [EntityInfo(k["x"], k["y"], k["z"], k["name"]) for k in data["entities"]]
            # checkEnts([ent.name for ent in entities], mobs_to_view)
            # # Find our closest cell, and work out what should be in it:
            # dist_to_cells = [abs(c[0] - current_x) + abs(c[1] - current_y) + abs(c[2] - current_z) for c in cell_midpoints]
            # target_ent_name = cell_midpoints[dist_to_cells.index(min(dist_to_cells))][3]
            # if mob_index < len(mobs_to_view) and target_ent_name == mobs_to_view[mob_index]:
            #     agent_host.sendCommand("chat hello " + target_ent_name)
            #     mob_index += 1
            # Attempt to find that entity in our entities list:
            # target_ents = [e for e in entities if e.name == target_ent_name]
            # if len(target_ents) != 1:
            #     pass
            # else:
            target_ent = entities[-1]
            print(entities)
            print(target_ent)
            # Look up height of entity from our table:
            target_height = [e for e in mob_list if e[0] == target_ent.name][0][2]
            # Calculate where to look in order to see it:
            target_yaw, target_pitch = calcYawAndPitchToMob(target_ent, current_x, current_y, current_z, target_height)
            # And point ourselves there:
            pointing_at_target = pointTo(agent_host, data, target_pitch, target_yaw, 0.5)
            agent_host.sendCommand('attack 1')
