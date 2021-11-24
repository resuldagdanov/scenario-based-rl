#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import sys
import os
  
#to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import carla

from examples.manual_control import (World, HUD, KeyboardControl, CameraManager,
                                     CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor)

import argparse
import logging
import time
import pygame

import torch as T
import matplotlib.pyplot as plt
from models.architecture.agent import Agent
import numpy as np
import random
import math

chkpt_dir = parent + "/models/saved_models" #where models are saved

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class WorldSR(World):

    restarted = False

    def restart(self):

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break
        
        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

def initialize_agent():
    random.seed(1)
    np.random.seed(1)

    is_cpu = True

    if is_cpu:
        device = 'cpu'
    else:
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    print("device ", device)

    input_dims = (30, 40, 3)  # env.observation_space.shape
    n_actions = 2  # env.action_space.shape[0]
    max_action = [1., 1.]  # env.action_space.high #todo: check the usage

    print(
        f"input_dims {input_dims}\nn_actions {n_actions}\nmax_action {max_action}")

    batch_size = 64
    buffer_size = 500_000
    agent = Agent(device, max_action, input_dims=input_dims, n_actions=n_actions,
                  max_size=buffer_size, batch_size=batch_size, chkpt_dir=chkpt_dir)

    return agent


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = WorldSR(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        agent = initialize_agent()
        print("agent initialized")

        num_episodes = 1
        max_step = 500 #1000
        for episode in range(num_episodes):
            print(f"episode {episode}")
            episode_reward = 0
            done = False

            step_num = 0
            while not done:
                clock.tick_busy_loop(60)
                #if controller.parse_events(client, world, clock):
                #    return

                state = np.zeros((30, 40, 3))
                action = agent.choose_action(state)

                steer = float(action[0])
                accel_brake = float(action[1])

                steer = steer * 0.5

                if accel_brake >= 0:
                    throttle = accel_brake
                    brake = 0.0
                else:
                    brake = abs(accel_brake)
                    throttle = 0.0

                controller._control.throttle = throttle
                controller._control.steer = steer
                controller._control.brake = brake
                world.player.apply_control(controller._control)
                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                next_state = np.zeros((30, 40, 3))

                velocity = world.player.get_velocity()
                kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
                reward = kmh*10

                print(f"step {step_num} action {action} reward {reward}")

                agent.remember(state, action, reward, next_state, done)
                agent.learn()

                #state = next_state
                episode_reward += reward

                step_num += 1

                #agent.save_models()

                """
                if step_num == max_step: #todo: add terminal criteria
                    done = True
                    
                    if (world and world.recording_enabled):
                        client.stop_recorder()

                    if world is not None:
                        world.destroy()

                    world.restart()
                """

        """
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock):
                return
            world.render(display)
            pygame.display.flip()
        """

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    args = argparser.parse_args()

    args.rolename = 'hero'      # Needed for CARLA version
    args.filter = "vehicle.*"   # Needed for CARLA version
    args.gamma = 2.2   # Needed for CARLA version
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
