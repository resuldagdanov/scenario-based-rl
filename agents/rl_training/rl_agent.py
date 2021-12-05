import carla
import pygame
import math
import os
import sys
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from _scenario_runner.manual_control import HUD, World

class RLAgent(object):
    def __init__(self, args, agent, resnet50_model):
        self.args = args
        self.is_terminal = False
        self.max_step = 200
        self.agent = agent
        self.resnet50_model = resnet50_model

    def set_terminal(self, status):
        self.is_terminal = status

    def initialize_agent_world(self):
        pygame.init()
        pygame.font.init()
        self.world = None

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(20.0)
        sim_world =  self.client.get_world()

        self.display = pygame.display.set_mode((self.args.width, self.args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(self.args.width, self.args.height)
        self.world = World(sim_world, hud, self.args)

        sim_world.wait_for_tick()

        self.clock = pygame.time.Clock()

    def run_one_episode(self):
        episode_reward = 0
        step_num = 0
        self.set_terminal(False)

        while True:
            self.clock.tick_busy_loop(60)

            #observation_np = np.zeros((3, 30, 40))
            if self.world != None and self.world.camera_manager != None and self.world.camera_manager.is_saved and self.world.camera_manager.rgb_data.any() != None:
                observation = self.world.camera_manager.rgb_data
                observation = np.reshape(observation, (3, self.args.height, self.args.width))
            else:
                #print("observation is none")
                observation = np.zeros((3, self.args.height, self.args.width))
            
            state = self.resnet50_model(observation)
            action = self.agent.choose_action(state)

            steer = float(action[0])
            accel_brake = float(action[1])

            steer = steer * 0.5

            if accel_brake >= 0:
                throttle = accel_brake
                brake = 0.0
            else:
                brake = abs(accel_brake)
                throttle = 0.0

            vc = carla.VehicleControl()
            vc.throttle = throttle #todo:change this
            vc.steer = steer
            vc.brake = brake
            self.world.player.apply_control(vc)
            self.world.tick(self.clock)

            #if not world.tick(clock):
            #    return

            self.world.render(self.display)
            pygame.display.flip()

            if self.world != None and self.world.camera_manager != None and self.world.camera_manager.is_saved and self.world.camera_manager.rgb_data.any() != None:
                next_observation = self.world.camera_manager.rgb_data
                next_observation = np.reshape(next_observation, (3, self.args.height, self.args.width))
            else:
                #print("next_observation is none")
                next_observation = np.zeros((3, self.args.height, self.args.width))

            next_state = self.resnet50_model(next_observation)

            velocity = self.world.player.get_velocity()
            kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
            if kmh == 0:
                reward = -10
            else:
                reward = kmh*10

            print(f"\tstep {step_num} action {action} reward {reward} is_terminal {self.is_terminal}")

            self.agent.remember(np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(self.is_terminal))
            self.agent.learn()

            episode_reward += reward
            #plot(agent)

            step_num += 1

            if self.is_terminal: #todo: when self.is_terminal turns into true here, the last step is not added to the agent's buffer, do it with if else above
                if self.args.save_model:
                    self.agent.save_models()
                break

            if step_num == self.max_step:
                self.set_terminal(True)
       
    def destroy_agent_world(self):
        if (self.world and self.world.recording_enabled):
             self.client.stop_recorder()

        if self.world is not None:
            # prevent destruction of ego vehicle
            if self.args.keep_ego_vehicle:
                self.world.player = None
            self.world.destroy()

        pygame.quit()