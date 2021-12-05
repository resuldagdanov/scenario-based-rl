from threading import Thread
import sys
import os
from types import SimpleNamespace
import argparse
from argparse import RawTextHelpFormatter
import torch as T
import numpy as np
import random
import time

#to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from networks.agent import Agent
from _scenario_runner.scenario_runner import ScenarioRunner
from rl_training.rl_agent import RLAgent

def function_scenario_runner(scenario_runner):
    scenario_runner.run()

def function_rlagent(rl_agent):
    rl_agent.initialize_agent_world()
    rl_agent.run_one_episode()
    rl_agent.destroy_agent_world()

def initialize_agent():
    random.seed(1) #todo: add seed to the args
    np.random.seed(1)

    is_cpu = True #todo:add this to the args

    if is_cpu:
        device = 'cpu' #T.device("cpu")
    else:
        device = 'cuda' if T.cuda.is_available() else 'cpu' #T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    print("device ", device)

    input_dims = (3, 30, 40)  # env.observation_space.shape
    n_actions = 2  # env.action_space.shape[0]
    max_action = [1., 1.]  # env.action_space.high #todo: check the usage

    print(f"input_dims {input_dims}\nn_actions {n_actions}\nmax_action {max_action}")

    batch_size = 64
    buffer_size = 500_000
    checkpoint_dir = parent + os.path.sep + "models"
    print(f"models will be saved to {checkpoint_dir}")
    agent = Agent(device, max_action, input_dims=input_dims, n_actions=n_actions,
                max_size=buffer_size, batch_size=batch_size, checkpoint_dir=checkpoint_dir)

    #agent.load_models()

    return agent

def main():
    parser = argparse.ArgumentParser(description="Main", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--scenario', default='StationaryObjectCrossing_1', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle (default: StationaryObjectCrossing_1)')
    parser.add_argument(
        '--max_episode', default=1, help='Number of episodes to train the agent (default: 1)')
    
    main_arguments = parser.parse_args()

    agent = initialize_agent()

    scenario = main_arguments.scenario
    max_episode = main_arguments.max_episode

    scenario_runner_args = SimpleNamespace(max_episode=max_episode, additionalScenario='', agent=None, agentConfig='', configFile='', debug=False, file=False, host='127.0.0.1', json=False, junit=False, list=False, openscenario=None, openscenarioparams=None, output=False, outputDir='', port='2000', randomize=False, record='', reloadWorld=True, repetitions=1, route=None, scenario=scenario, sync=False, timeout='10.0', trafficManagerPort='8000', trafficManagerSeed='0', waitForEgo=False) #todo: understand reloadWorld and repetitions arguments
    scenario_runner = ScenarioRunner(scenario_runner_args)

    rl_agent_args = SimpleNamespace(max_episode=max_episode, autopilot=False, debug=False, height=720, host='127.0.0.1', keep_ego_vehicle=False, port=2000, res='1280x720', rolename='hero', width=1280)
    rl_agent = RLAgent(rl_agent_args, agent)

    for episode in range(int(main_arguments.max_episode)):
        scenario_runner_thread = Thread(target=function_scenario_runner, args=(scenario_runner,))
        rl_agent_thread = Thread(target=function_rlagent, args=(rl_agent,))
        
        scenario_runner_thread.start()
        rl_agent_thread.start()

        scenario_runner_thread.join()
        rl_agent.set_terminal(True)
        rl_agent_thread.join()
        print(f"End of episode {episode}")

if __name__ == "__main__":
    main()