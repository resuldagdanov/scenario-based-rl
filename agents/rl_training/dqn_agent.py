from __future__ import print_function

import os
import signal
import sys
import numpy as np
import carla
import torch as T
import cv2
import math
import weakref
import random

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from agent_utils import base_utils
from agent_utils.pid_controller import PIDController
from agent_utils.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent


# TODO: sensor configs can be put to DB
SENSOR_CONFIG = {
            'width': 400,
            'height': 300,
            'fov': 100
        }

WEATHERS = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
        'ClearSunset': carla.WeatherParameters.ClearSunset,

        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
        'CloudySunset': carla.WeatherParameters.CloudySunset,

        'WetNoon': carla.WeatherParameters.WetNoon,
        'WetSunset': carla.WeatherParameters.WetSunset,

        'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
        'MidRainSunset': carla.WeatherParameters.MidRainSunset,

        'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
        'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,

        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
        'HardRainSunset': carla.WeatherParameters.HardRainSunset,

        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
        'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}
WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return 'DqnAgent' 


class DqnAgent(autonomous_agent.AutonomousAgent):
    def setup(self,  path_to_conf_file, route_id, rl_model):
        self.track = autonomous_agent.Track.SENSORS 

        self.agent = rl_model

        self.debug = self.agent.debug
        self.writer = self.agent.writer

        if not self.agent.evaluate:
            self.best_reward = self.agent.db.get_best_reward(self.agent.training_id)
            self.total_step_num = self.agent.db.get_total_step_num(self.agent.training_id)
            self.epsilon = self.agent.db.get_epsilon(self.agent.training_id)
            print(f"training model_name {self.agent.model_name}   episode_number {self.agent.db.get_global_episode_number(self.agent.training_id)}   total_step_num {self.total_step_num}   latest_sample_id {self.agent.db.get_latest_sample_id(self.agent.training_id)}   best_reward {self.best_reward}   best_reward_episode_number {self.agent.db.get_best_reward_episode_number(self.agent.training_id)}")
        else:
            self.best_reward = 0.0
            self.total_step_num = self.agent.db.get_evaluation_total_step_num(self.agent.evaluation_id)
            print(f"evaluation model_name {self.agent.model_name}   model_episode_number {self.agent.db.get_evaluation_model_episode_number(self.agent.evaluation_id)}   episode_number {self.agent.db.get_evaluation_global_episode_number(self.agent.evaluation_id)}   total_step_num {self.total_step_num}   average_evaluation_score {self.agent.db.get_evaluation_average_evaluation_score(self.agent.evaluation_id)}")

        self.initialized = False
        self._sensor_data = SENSOR_CONFIG
        self.weather_id = WEATHERS_IDS[0]

    def init_dnn_agent(self):
        input_dims = (3, SENSOR_CONFIG['height'], SENSOR_CONFIG['width'])
        self.device = self.agent.device

    def init_auto_pilot(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner.set_route(self._global_plan, True)

        self.init_dnn_agent()
        self.init_auto_pilot()
        
        self.init_privileged_agent()

        self.initialized = True
        self.push_buffer = False

        self.next_image_features = []
        self.next_fused_inputs = []

        self.step_number = 1
        self.episode_total_reward = 0.0
        self.count_vehicle_stop = 0
        self.count_is_seen = 0
        self.n_updates = 0
        self.total_loss = 0.0

        self.previous_gps = 0.0

        self.is_autopilot = True
        self.autopilot_counter = 0
        self.constant_action = None

        self.autopilot_step = random.randint(95, 105) # red light # random.randint(98, 128) # stuck vehicle
        print("\nRandom Autopilot Step: ", self.autopilot_step)

        if self.debug:
            cv2.namedWindow("RL-rgb-front")

    def get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def init_privileged_agent(self):
        self.hero_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.hero_vehicle.get_world()

        self.collision_intensity = 0.0
        self.is_collision = False
        # self.is_lane_invasion = False

        self.privileged_sensors()

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'rgb_front'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]   
    
    def tick(self, input_data):
        rgb_front = input_data['rgb_front'][1][:, :, :3]
        rgb_front = rgb_front[:, :, ::-1]
        gps = input_data['gps'][1][:2]
        compass = input_data['imu'][1][-1]
        speed = input_data['speed'][1]['speed']

        return {
            'rgb_front': rgb_front,
            'gps': gps,
            'speed': speed,
            'compass': compass
            }

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.autopilot_counter += 1

        data = self.tick(input_data)
        gps = self.get_position(data)
        speed = data['speed']
        compass = data['compass']

        if self.autopilot_counter % 10 == 0:
            self.previous_gps = gps

            if not self.agent.evaluate:
                self.change_weather()

        near_node, near_command = self._route_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        # front image
        rgb_front_image = data['rgb_front']
        front_cv_image = rgb_front_image[:, :, ::-1]

        fused_inputs = np.zeros(3, dtype=np.float32)

        fused_inputs[0] = speed
        fused_inputs[1] = far_node[0] - gps[0]
        fused_inputs[2] = far_node[1] - gps[1]

        if self.debug:
            disp_front_image = cv2.UMat(front_cv_image)
            cv2.imshow("RL-rgb-front", disp_front_image)
            cv2.waitKey(1)

        # construct network input image format
        dnn_input_image = self.image_to_dnn_input(image=front_cv_image)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = T.from_numpy(fused_inputs.copy()).unsqueeze(0).to(self.device)

        # apply freezed pre-trained resnet model onto the image
        with T.no_grad():
            image_features_torch = self.agent.resnet50(dnn_input_image)
            image_features = image_features_torch.cpu().detach().numpy()[0]

        # get action from value network
        if self.agent.evaluate: # evaluation
            dnn_agent_action = int(self.agent.select_max_action(image_features=image_features_torch))          
        else: # training
            dnn_agent_action = int(self.agent.select_action(image_features=image_features_torch, epsilon=self.epsilon))
        
        if self.autopilot_counter > self.autopilot_step + 50:
            self.is_autopilot = False

        if self.autopilot_counter > self.autopilot_step and self.is_autopilot is True:
            dnn_agent_action = 0
        elif self.autopilot_counter <= self.autopilot_step and self.is_autopilot is True:
            dnn_agent_action = 2
        else:
            dnn_agent_action = dnn_agent_action
        
        throttle, steer, brake, angle = self.calculate_high_level_action(dnn_agent_action, compass, gps, near_node, far_node, data)
        
        applied_control = carla.VehicleControl()
        applied_control.throttle = throttle
        applied_control.steer = steer
        applied_control.brake = brake

        # compute step reward and deside for termination
        if self.is_autopilot is False:
            reward, done = self.calculate_reward(throttle=throttle, ego_speed=speed, ego_gps=gps)

        #self.is_lane_invasion = False
        self.is_collision = False

        if self.is_autopilot is False:
            loss = None

            if self.push_buffer and not self.agent.evaluate:
                self.agent.memory.push(image_features, fused_inputs, dnn_agent_action, reward, self.next_image_features, self.next_fused_inputs, done)

                if self.agent.memory.filled_size > self.agent.batch_size:
                    sample_batch = self.agent.memory.sample(self.agent.batch_size)

                    loss = self.agent.update(sample_batch)
                
                    self.n_updates += 1 #number of updates in one episode
                    self.total_loss += loss #episodic loss

            self.next_image_features = image_features
            self.next_fused_inputs = fused_inputs
            
            self.push_buffer = True

            self.episode_total_reward += reward

            if not self.agent.evaluate: #training
                self.agent.db.insert_data_to_reward_table(self.agent.db.get_model_name(self.agent.training_id), self.agent.db.get_global_episode_number(self.agent.training_id), self.step_number, reward)
                # running_average = self.agent.db.get_running_average(self.agent.db.get_model_name(self.agent.training_id))
                # base_utils.tensorboard_writer_running_average(self.writer, self.total_step_num, running_average)

                if loss is not None:
                    print("[Action]: epsilon: {:.2f}, high_level_action: {:d}, throttle: {:.2f}, steer: {:.2f}, brake: {:.2f}, speed: {:.2f}kmph, loss: {:.2f}, reward: {:.2f}, step: #{:d}, total_step: #{:d}".format(self.epsilon, dnn_agent_action, throttle, steer, brake, speed * 3.6, loss, reward, self.step_number, self.total_step_num))
                    self.agent.db.insert_data_to_loss_table(self.agent.db.get_model_name(self.agent.training_id), self.agent.db.get_global_episode_number(self.agent.training_id), self.step_number, float(loss))
                else:
                    print("[Action]: epsilon: {:.2f}, high_level_action: {:d}, throttle: {:.2f}, steer: {:.2f}, brake: {:.2f}, speed: {:.2f}kmph, reward: {:.2f}, step: #{:d}, total_step: #{:d}".format(self.epsilon, dnn_agent_action, throttle, steer, brake, speed * 3.6, reward, self.step_number, self.total_step_num))
            else: #evaluation
                print("[Action]: high_level_action: {:d}, throttle: {:.2f}, steer: {:.2f}, brake: {:.2f}, speed: {:.2f}kmph, reward: {:.2f}, step: #{:d}, total_step: #{:d}".format(dnn_agent_action, throttle, steer, brake, speed * 3.6, reward, self.step_number, self.total_step_num))

            if not self.agent.evaluate:
                if self.total_step_num % 1000 == 0: # TODO: Make this hyperparam
                    self.agent.target_update()

            # terminate an episode
            if done:
                if not self.agent.evaluate: #training
                    self.epsilon *= self.agent.epsilon_decay # exponential decay
                    self.epsilon = max(self.epsilon, self.agent.epsilon_min)
                    self.agent.db.update_epsilon(self.epsilon, self.agent.training_id)

                    self.agent.db.update_latest_sample_id(self.agent.memory.id, self.agent.training_id)
                    self.agent.db.update_total_step_num(self.total_step_num, self.agent.training_id)

                    if self.episode_total_reward > self.best_reward:
                        self.best_reward = self.episode_total_reward
                        self.agent.db.update_best_reward(self.best_reward, self.agent.training_id)
                        self.agent.db.update_best_reward_episode_number(self.agent.db.get_global_episode_number(self.agent.training_id), self.agent.training_id) # TODO: make sure that global episode number is correct

                        print("Best Episode Reward: ", self.best_reward)

                        self.agent.save_models(self.agent.db.get_global_episode_number(self.agent.training_id))

                    base_utils.tensorboard_writer_with_one_loss(self.writer, self.agent.db.get_global_episode_number(self.agent.training_id), self.episode_total_reward, self.best_reward, self.total_loss, self.n_updates)            
                else: #evaluation
                    self.agent.db.update_evaluation_total_step_num(self.total_step_num, self.agent.evaluation_id)
                    self.agent.db.update_evaluation_average_evaluation_score(self.episode_total_reward, self.agent.evaluation_id)
                    base_utils.tensorboard_writer_evaluation(self.writer, self.agent.db.get_evaluation_global_episode_number(self.agent.evaluation_id), self.episode_total_reward)

                print("------------ Terminating! ------------")
                print("Episode Total Reward: ", round(self.episode_total_reward, 3))
                self.destroy()
        
        if self.is_autopilot is False:
            self.step_number += 1
            self.total_step_num += 1

        return applied_control


    def shift_point(self, ego_compass, ego_gps, near_node, offset_amount):
        # rotation matrix
        R = np.array([
            [np.cos(np.pi / 2 + ego_compass), -np.sin(np.pi / 2 + ego_compass)],
            [np.sin(np.pi / 2 + ego_compass), np.cos(np.pi / 2 + ego_compass)]
        ])

        # transpose of rotation matrix
        trans_R = R.T

        local_command_point = np.array([near_node[0] - ego_gps[0], near_node[1] - ego_gps[1]])
        local_command_point = trans_R.dot(local_command_point)

        # positive offset shifts near node to right; negative offset shifts near node to left
        local_command_point[0] += offset_amount
        local_command_point[1] += 0

        new_near_node = np.linalg.inv(trans_R).dot(local_command_point)

        new_near_node[0] += ego_gps[0]
        new_near_node[1] += ego_gps[1]

        return new_near_node

    def calculate_high_level_action(self, high_level_action, compass, gps, near_node, far_node, data):
        #0 brake
        #1 no brake - go to left lane of next_waypoint
        #2 no brake - keep lane (stay at next_waypoint's lane)
        #3 no brake - go to right lane of next_waypoint

        if high_level_action == 1: # left
            offset = -3.5
            new_near_node = self.shift_point(ego_compass=compass, ego_gps=gps, near_node=near_node, offset_amount=offset)
        
        elif high_level_action == 3: # right
            offset = 3.5
            new_near_node = self.shift_point(ego_compass=compass, ego_gps=gps, near_node=near_node, offset_amount=offset)
        
        else: # keep lane
            offset = 0.0
            new_near_node = near_node

        # get auto-pilot actions
        steer, throttle, target_speed, angle = self.get_control(new_near_node, far_node, data)

        if high_level_action == 0: # brake
            throttle = 0.0
            brake = 1.0
        else: # no brake
            throttle = throttle
            brake = 0.0

        return throttle, steer, brake, angle

    def calculate_reward(self, throttle, ego_speed, ego_gps):
        reward = 0.0
        done = 0

        # if any of the following is not None, then the agent should brake
        is_light, is_walker, is_vehicle, _ = self.traffic_data()
        print("[Traffic]: traffic light-", is_light, " walker-", is_walker, " vehicle-", is_vehicle)

        # give penalty if ego vehicle is not braking where it should brake
        if any(x is not None for x in [is_light, is_walker, is_vehicle]):
            
            # accelerating while it should brake
            if throttle < 0.1:
                print("[Reward]: correctly braking !")
                # braking is desired
                reward += 50
            else:
                print("[Penalty]: not braking !")

            reward -= ego_speed

        else:
            reward += ego_speed
        
        # distance from starting position
        reward += np.linalg.norm(ego_gps - self.previous_gps)

        # TODO: make this hyperparam
        if self.step_number > 250:
            done = 1

        # negative reward for collision
        if self.is_collision:
            print("[Penalty]: collision !")
            reward -= 1500
            done = 1

        return reward, done
    
    def privileged_sensors(self):
        blueprint = self.world.get_blueprint_library()

        # get blueprint of the sensors
        bp_collision = blueprint.find('sensor.other.collision')
        # bp_lane_invasion = blueprint.find('sensor.other.lane_invasion')

        # attach sensors to the ego vehicle
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)
        # self.lane_invasion_sensor = self.world.spawn_actor(bp_lane_invasion, carla.Transform(), attach_to=self.hero_vehicle)

        # create sensor event callbacks
        self.collision_sensor.listen(lambda event: DqnAgent._on_collision(weakref.ref(self), event))
        # self.lane_invasion_sensor.listen(lambda event: DqnAgent._on_lane_invasion(weakref.ref(self), event))

    def traffic_data(self):
        all_actors = self.world.get_actors()

        lights_list = all_actors.filter('*traffic_light*')
        walkers_list = all_actors.filter('*walker*')
        vehicle_list = all_actors.filter('*vehicle*')
        stop_list = all_actors.filter('*stop*')

        traffic_lights = base_utils.get_nearby_lights(self.hero_vehicle, lights_list)
        stops = base_utils.get_nearby_lights(self.hero_vehicle, stop_list) # TODO: if you need different radius, write new function

        light = self.is_light_red(traffic_lights)
        walker = self.is_walker_hazard(walkers_list)
        vehicle = self.is_vehicle_hazard(vehicle_list)

        if len(stops) == 0:
            stop = None
        else:
            stop = stops

        return light, walker, vehicle, stop

    def get_control(self, target, far_target, tick_data):
        pos = self.get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # steering
        angle_unnorm = self.get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # acceleration
        angle_far_unnorm = self.get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4.0 if should_slow else 7.0
        
        self.should_slow = int(should_slow)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm
        
        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        return steer, throttle, target_speed, angle

    def get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle 

        return angle

    def image_to_dnn_input(self, image):
        # convert width height channel to channel width height
        image = np.array(image.transpose((2, 0, 1)), np.float32)
        # BGRA to BGR
        image = image[:3, :, :]
        # BGR to RGB
        image = image[::-1, :, :]
        # normalize to 0 - 1
        image = image / 255
        # convert image to torch tensor
        image = T.from_numpy(image.copy()).unsqueeze(0)
        
        # normalize input image (using default torch normalization technique)
        image = self.normalize_rgb(image)
        image = image.to(self.device)

        return image

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def is_light_red(self, traffic_lights):
        for light in traffic_lights:
            if light.get_state() == carla.TrafficLightState.Red:
                return True
            elif light.get_state() == carla.TrafficLightState.Yellow:
                return True
        
        return None

    def is_walker_hazard(self, walkers_list):
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        v1 = 13.0 * base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        
        for walker in walkers_list:
            v2_hat = base_utils._orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(base_utils._numpy(walker.get_velocity()))
            
            if s2 < 0.05:
                v2_hat *= s2
            
            p2 = -3.0 * v2_hat + base_utils._numpy(walker.get_location())
            v2 = 10.0 * v2_hat
            
            collides, collision_point = base_utils.get_collision(p1, v1, p2, v2)
            
            if collides:
                return walker

        return None

    def is_vehicle_hazard(self, vehicle_list):
        o1 = base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        s1 = max(10, 5.0 * np.linalg.norm(base_utils._numpy(self.hero_vehicle.get_velocity()))) # increases the threshold distance
        s2 = max(20, 5.0 * np.linalg.norm(base_utils._numpy(self.hero_vehicle.get_velocity()))) # increases the threshold distance
        
        v1_hat = o1
        v1 = s1 * v1_hat
        
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self.hero_vehicle.id:
                continue
            
            o2 = base_utils._orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = base_utils._numpy(target_vehicle.get_location())
            s2 = max(6.0, 4.0 * np.linalg.norm(base_utils._numpy(target_vehicle.get_velocity())))
            
            v2_hat = o2
            v2 = s2 * v2_hat
            p2_p1 = p2 - p1
            
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)
            
            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
            
            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1 and distance < s2:
                self.target_vehicle_speed = target_vehicle.get_velocity()
                continue
            elif distance > s1:
                continue
            return target_vehicle
        
        return None

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.is_collision = True

        impulse = event.normal_impulse
        self.collision_intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

    # @staticmethod
    # def _on_lane_invasion(weak_self, event):
    #     self = weak_self()
    #     if not self:
    #         return

    #     self.is_lane_invasion = True 

    def change_weather(self):
        index = random.choice(range(len(WEATHERS)))

        self.weather_id = WEATHERS_IDS[index]
        self.world.set_weather(WEATHERS[WEATHERS_IDS[index]])

    def destroy(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()

        # if self.lane_invasion_sensor is not None:
        #     self.lane_invasion_sensor.stop()
        
        # terminate and go to another episode
        os.kill(os.getpid(), signal.SIGINT)