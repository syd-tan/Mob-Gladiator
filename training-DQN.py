from DQN import SAVE_CHECKPOINT_FREQ, TARGET_UPDATE_FREQ, DQN
import torch
import gladiator
from initialize_mission import initialize_mission, get_mob
import MalmoPython
import malmoutils
import time
import lookatmob
from datetime import datetime, timedelta

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
    agent = DQN(12, 8)
    agent_host = MalmoPython.AgentHost()
    malmoutils.fix_print()
    malmoutils.parse_command_line(agent_host)
    recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
    video_requirements = '<VideoProducer><Width>860</Width><Height>480</Height></VideoProducer>' if agent_host.receivedArgument("record_video") else ''

    # Add Minecraft Client
    my_client_pool = MalmoPython.ClientPool()
    my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

    msPerTick = 50  # 50 ms. per tick is default

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
            "Gladiator Begin! #" + str(iRepeat), msPerTick, video_requirements
        )

        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        enemy_mob = get_mob()
        # Set up a recording
        my_mission_record = MalmoPython.MissionRecordSpec()
        if recordingsDirectory:
            if agent_host.receivedArgument("record_video"):
                my_mission_record.recordMP4(24,2000000)
            my_mission_record.setDestination(recordingsDirectory + f"/{enemy_mob}" + "//" + "Mission_" + str(iRepeat) + ".tgz")

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

        # initializes mission
        world_state = initialize_mission(agent_host, enemy_mob)

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
        finish_time = datetime.now() + timedelta(seconds=30)
        curr_state, _, _ = gladiator.step(
            agent_host, world_state, curr_state, enemy_mob, finish_time
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
                next_state, step_reward, done = gladiator.step(agent_host, world_state, curr_state, enemy_mob, finish_time)
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

        # mission has ended.
        for error in world_state.errors:
            print("Error:", error.text)
        # Evaluate: Change episode reward in array and add reward/attacks to records for that mob
        agent.rewards_per_monster[enemy_mob].append(total_reward)
        agent.attacks_per_monster[enemy_mob].append(total_attacks)
        agent.remaining_agent_hp_per_monster[enemy_mob].append(curr_state.agent_health if curr_state.agent_health > 0 else 0)
        agent.winrates_per_monster[enemy_mob] = (
            agent.winrates_per_monster[enemy_mob][0] + (1 if curr_state.agent_health > 0 and curr_state.enemy_health == 0 else 0),
            agent.winrates_per_monster[enemy_mob][1] + 1
        )

        if iRepeat % SAVE_CHECKPOINT_FREQ == 0:
            agent.save_checkpoint(iRepeat)
        if iRepeat % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        print()
        print("=" * 41)
        print("Total score this round:", total_reward)
        print("=" * 41)
        print()
        time.sleep(1)  # Give the mod a little time to prepare for the next mission.