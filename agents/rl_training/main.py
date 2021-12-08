from threading import Thread
import sys
import os
from types import SimpleNamespace
import argparse
from argparse import RawTextHelpFormatter
import torch as T
import numpy as np
import random
import traceback
from tensorboardX import SummaryWriter

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from networks.networks import RESNET50Model
from networks.agent import Agent
from _scenario_runner.scenario_runner import ScenarioRunner
from rl_training.rl_agent import RLAgent


def function_scenario_runner(scenario_runner):
    try:
        scenario_runner.run()

    except Exception:
        traceback.print_exc()


def function_rl_agent(rl_agent):
    try:
        rl_agent.initialize_agent_world()
        rl_agent.run_one_episode()
        rl_agent.destroy_agent_world()

    except Exception:
        traceback.print_exc()


def initialize_agent(args, device):
    input_dims = (3, args.height, args.width)
    n_actions = 2
    max_action = [1., 1.]

    print(f"input_dims: {input_dims}\nn_actions: {n_actions}\nmax_action: {max_action}")

    checkpoint_dir = parent + os.path.sep + "models"
    print(f"models will be saved to {checkpoint_dir}")

    agent = Agent(device=device, max_action=max_action, n_actions=n_actions, max_size=args.buffer_size, batch_size=args.batch_size, checkpoint_dir=checkpoint_dir)

    resnet50_model = RESNET50Model(device=device, input_dims=input_dims, checkpoint_dir=checkpoint_dir)

    if args.load_model:
        agent.load_models()

    return agent, resnet50_model


def main():
    parser = argparse.ArgumentParser(description="Main Arguments for RL Training on Specific Scenario", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--scenario', default='StationaryObjectCrossing_1', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle (default: StationaryObjectCrossing_1)')
    parser.add_argument(
        '--max_episode', default=5, help='Number of episodes to train the agent (default: 5)', type=int)
    parser.add_argument(
        '--seed', default=1, help='Seed for random and numpy packages (default: 1)', type=int)
    parser.add_argument(
        '--cpu', default=True, help='true=CPU false=CUDA (default is True)')
    parser.add_argument(
        '--batch_size', default=32, help='Batch size for RL Agent (default: 32)', type=int)
    parser.add_argument(
        '--buffer_size', default=500_000, help='Buffer size for RL Agent (default: 500_000)', type=int)
    parser.add_argument(
        '--load_model', default=False, help='Load saved models for RL Agent (default is False)')
    parser.add_argument(
        '--save_model', default=True, help='Save models of RL Agent (default is True)')
    parser.add_argument(
        '--height', default=300, help='Camera height (default: 720)', type=int)
    parser.add_argument(
        '--width', default=400, help='Camera width (default: 1280)', type=int)
    parser.add_argument(    # TODO: use default parameters for only pygame visualization
        '--height_pygame', default=300, help='Camera height (default: 720)', type=int)
    parser.add_argument(
        '--width_pygame', default=400, help='Camera width (default: 1280)', type=int)
    
    main_arguments = parser.parse_args()

    for arg in vars(main_arguments):
        print(f"{arg}: {getattr(main_arguments, arg)}")

    random.seed(main_arguments.seed)
    np.random.seed(main_arguments.seed)

    if main_arguments.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if T.cuda.is_available() else 'cpu'
    print("device: ", device)

    network_res = str(main_arguments.width) + "x" + str(main_arguments.height)
    pygame_res = str(main_arguments.width_pygame) + "x" + str(main_arguments.height_pygame)
    print(f"network input resolution: {network_res}  -- pygame HUD resolution: {pygame_res}")

    agent, resnet50_model = initialize_agent(args=main_arguments, device=device)
    print("\n")

    writer = SummaryWriter(comment="_model")

    total_reward = []
    total_average_reward = 0.0
    total_steps = 0

    scenario_runner_args = SimpleNamespace(
                                max_episode=main_arguments.max_episode,
                                additionalScenario='',
                                agent=None,
                                agentConfig='',
                                configFile='',
                                debug=False,
                                file=False,
                                host='127.0.0.1',
                                json=False,
                                junit=False,
                                list=False,
                                openscenario=None,
                                openscenarioparams=None,
                                output=False,
                                outputDir='',
                                port='2000',
                                randomize=False,
                                record='',
                                reloadWorld=True,
                                repetitions=1,
                                route=None,
                                scenario=main_arguments.scenario,
                                sync=False,
                                timeout='20.0',
                                trafficManagerPort='8000',
                                trafficManagerSeed='0',
                                waitForEgo=False)

    rl_agent_args = SimpleNamespace(
                                batch_size=main_arguments.batch_size,
                                max_episode=main_arguments.max_episode,
                                save_model=main_arguments.save_model,
                                autopilot=False,
                                debug=False,
                                host='127.0.0.1',
                                keep_ego_vehicle=False,
                                port=2000,
                                timeout='20.0',
                                rolename='hero',
                                network_res=network_res,
                                pygame_res=pygame_res,
                                width=main_arguments.width,
                                height=main_arguments.height,
                                width_pygame=main_arguments.width_pygame,
                                height_pygame=main_arguments.height_pygame)


    scenario_runner = ScenarioRunner(
                    args=scenario_runner_args)

    rl_agent = RLAgent(
                    args=rl_agent_args,
                    agent=agent,
                    resnet50_model=resnet50_model,
                    writer=writer,
                    total_reward=total_reward,
                    total_average_reward=total_average_reward,
                    total_steps=total_steps)

    # loop for running each episode
    for episode in range(main_arguments.max_episode):

        scenario_runner_thread = Thread(target=function_scenario_runner, args=(scenario_runner,))
        rl_agent_thread = Thread(target=function_rl_agent, args=(rl_agent,))
        
        rl_agent.set_episode(episode)
        
        scenario_runner_thread.start()
        rl_agent_thread.start()

        scenario_runner_thread.join()
        rl_agent.set_terminal(True)
        rl_agent_thread.join()

        print(f"End of episode {episode} buffer is filled with {agent.memory.mem_cntr} samples")

if __name__ == "__main__":
    main()