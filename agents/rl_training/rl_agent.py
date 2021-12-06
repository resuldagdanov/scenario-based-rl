import carla
import pygame
import math
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from _scenario_runner.manual_control import HUD, World

class RLAgent(object):
    def __init__(self, args, agent, resnet50_model, writer, total_reward, total_average_reward, total_steps):
        self.args = args
        self.is_terminal = False
        self.max_step = 200
        self.agent = agent
        self.resnet50_model = resnet50_model
        self.episode = 0
        self.writer = writer
        self.total_reward = total_reward
        self.total_average_reward = total_average_reward
        self.total_steps = total_steps

    def set_episode(self, episode):
        self.episode = episode

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

        self.hud = HUD(self.args.width, self.args.height)
        self.world = World(sim_world, self.hud, self.args)

        sim_world.wait_for_tick()

        self.clock = pygame.time.Clock()

    def tensorboard_writer(self, step_number, total_average_reward, average_reward, episode_reward):
        self.writer.add_scalar("step number - episode" , step_number, self.episode+1)
        self.writer.add_scalar("episode reward", episode_reward, self.episode+1)
        self.writer.add_scalar("total average reward - episode", total_average_reward, self.episode+1)
        self.writer.add_scalar("average reward - episode", average_reward, self.episode+1)
        if self.total_steps > int(self.args.batch_size):
            self.writer.add_scalar("actor loss - steps", self.agent.actor_losses[len(self.agent.actor_losses)-1], self.total_steps+1)
            self.writer.add_scalar("critic loss - steps", self.agent.critic_losses[len(self.agent.critic_losses)-1], self.total_steps+1)
            self.writer.add_scalar("value loss - steps", self.agent.value_losses[len(self.agent.critic_losses)-1], self.total_steps+1)

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

            if self.hud.is_colhist_saved:
                reward -= self.hud.collision[len(self.hud.collision)-1] * 100

            print(f"\tstep {step_num} action {action} reward {reward} is_terminal {self.is_terminal}")

            state = state.cpu().detach().numpy()[0]
            next_state = next_state.cpu().detach().numpy()[0]
            self.agent.remember(np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(self.is_terminal))
            self.agent.learn()
            #self.plot()

            episode_reward += reward
            #plot(agent)

            step_num += 1
            self.total_steps += 1

            if self.is_terminal: #todo: when self.is_terminal turns into true here, the last step is not added to the agent's buffer, do it with if else above
                if self.args.save_model:
                    self.agent.save_models()
                break

            if step_num == self.max_step:
                self.set_terminal(True)

        self.total_average_reward = self.average_calculation(self.total_average_reward, self.episode+1, episode_reward)        
        self.total_reward.append(episode_reward)
        average_reward = np.mean(self.total_reward[-20:])

        self.tensorboard_writer(step_num, self.total_average_reward, average_reward, episode_reward)
       
    def average_calculation(self, prev_avg, num_episodes, new_val):
        total = prev_avg * (num_episodes - 1)
        total = total + new_val
        return np.float(total / num_episodes)

    def destroy_agent_world(self):
        if (self.world and self.world.recording_enabled):
             self.client.stop_recorder()

        if self.world is not None:
            # prevent destruction of ego vehicle
            if self.args.keep_ego_vehicle:
                self.world.player = None
            self.world.destroy()

        pygame.quit()