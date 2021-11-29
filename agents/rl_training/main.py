import threading
import sys
import os
from types import SimpleNamespace
  
#to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from _scenario_runner.scenario_runner import ScenarioRunner
from rl_training.rl_agent import RLAgent

class ScenarioRunnerThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        arguments = self.get_arguments()
        self.scenario_runner = ScenarioRunner(arguments)

    def get_arguments(self):
        scenario = 'FollowLeadingVehicle_1'
        return SimpleNamespace(additionalScenario='', agent=None, agentConfig='', configFile='', debug=False, file=False, host='127.0.0.1', json=False, junit=False, list=False, openscenario=None, openscenarioparams=None, output=False, outputDir='', port='2000', randomize=False, record='', reloadWorld=True, repetitions=1, route=None, scenario=scenario, sync=False, timeout='10.0', trafficManagerPort='8000', trafficManagerSeed='0', waitForEgo=False)

    def run(self):
        #print ("Starting " + self.name)
        try:
            result = self.scenario_runner.run()
            #print(f"result {result}")
        except Exception as error:
            print(error)
        
        """
        try:
            self.scenario_runner.run()
        finally:
            if scenario_runner is not None:
                scenario_runner.destroy()
                del scenario_runner
        """
        #print ("End of run of " + self.name)

class RLAgentThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        arguments = self.get_arguments()
        self.rl_agent = RLAgent(arguments)

    def get_arguments(self):
        return SimpleNamespace(autopilot=False, debug=False, height=720, host='127.0.0.1', keep_ego_vehicle=False, port=2000, res='1280x720', rolename='hero', width=1280)

    def run(self):
        print ("Starting " + self.name)
        try:
            self.rl_agent.game_loop()
        except Exception as error:
            print(error)
        #print ("End of run of " + self.name)

scenarioRunnerThread = ScenarioRunnerThread(1, "ScenarioRunnerThread")
RLagentThread = RLAgentThread(2, "RLAgentThread")

scenarioRunnerThread.start()
RLagentThread.start()

scenarioRunnerThread.join() #wait for scenarioRunnerThread to stop
#print("scenarioRunnerThread.join()")
RLagentThread.rl_agent.set_terminal(True)

#print ("Exiting Main Thread")