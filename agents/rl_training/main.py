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

#to add the parent "agents" folder to sys path and import models
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
    input_dims = (3, 30, 40)
    n_actions = 2
    max_action = [1., 1.]

    print(f"input_dims: {input_dims}\nn_actions: {n_actions}\nmax_action: {max_action}")

    batch_size = args.batch_size
    buffer_size = args.buffer_size
    checkpoint_dir = parent + os.path.sep + "models"
    print(f"models will be saved to {checkpoint_dir}")

    resnet50_model = RESNET50Model(device, input_dims, checkpoint_dir=checkpoint_dir)

    agent = Agent(device, max_action, n_actions=n_actions, max_size=buffer_size, batch_size=batch_size, checkpoint_dir=checkpoint_dir)

    if args.load_model:
        agent.load_models()

    return agent, resnet50_model

def main():
    parser = argparse.ArgumentParser(description="Main", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--scenario', default='StationaryObjectCrossing_1', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle (default: StationaryObjectCrossing_1)')
    parser.add_argument(
        '--max_episode', default=2, help='Number of episodes to train the agent (default: 2)', type=int)
    parser.add_argument(
        '--seed', default=1, help='Seed for random and numpy packages (default: 1)', type=int)
    parser.add_argument(
        '--cpu', help='true=CPU false=CUDA (default is True)', action='store_false')
    parser.add_argument(
        '--batch_size', default=64, help='Batch size for RL Agent (default: 64)', type=int)
    parser.add_argument(
        '--buffer_size', default=500_000, help='Buffer size for RL Agent (default: 500_000)', type=int)
    parser.add_argument(
         '--load_model', help='Load saved models for RL Agent (default is False)', action='store_true')
    parser.add_argument(
         '--save_model', help='Save models of RL Agent (default is True)', action='store_false')
    
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

    agent, resnet50_model = initialize_agent(main_arguments, device)
    print("\n\n")

    scenario_runner_args = SimpleNamespace(max_episode=main_arguments.max_episode, additionalScenario='', agent=None, agentConfig='', configFile='', debug=False, file=False, host='127.0.0.1', json=False, junit=False, list=False, openscenario=None, openscenarioparams=None, output=False, outputDir='', port='2000', randomize=False, record='', reloadWorld=True, repetitions=1, route=None, scenario=main_arguments.scenario, sync=False, timeout='10.0', trafficManagerPort='8000', trafficManagerSeed='0', waitForEgo=False)
    scenario_runner = ScenarioRunner(scenario_runner_args)

    rl_agent_args = SimpleNamespace(max_episode=main_arguments.max_episode, save_model=main_arguments.save_model, autopilot=False, debug=False, height=720, host='127.0.0.1', keep_ego_vehicle=False, port=2000, res='1280x720', rolename='hero', width=1280)
    rl_agent = RLAgent(rl_agent_args, agent, resnet50_model)

    for episode in range(main_arguments.max_episode):
        scenario_runner_thread = Thread(target=function_scenario_runner, args=(scenario_runner,))
        rl_agent_thread = Thread(target=function_rl_agent, args=(rl_agent,))
        
        scenario_runner_thread.start()
        rl_agent_thread.start()

        scenario_runner_thread.join()
        rl_agent.set_terminal(True)
        rl_agent_thread.join()
        print(f"End of episode {episode} buffer is filled with {agent.memory.mem_cntr} samples")

if __name__ == "__main__":
    main()