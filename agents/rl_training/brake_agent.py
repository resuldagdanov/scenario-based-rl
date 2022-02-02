from __future__ import print_function

import os
import sys
import cv2
import math
import carla
import signal

import torch as T
import numpy as np
import random

"""
seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
"""

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.autoagents.autonomous_agent import AutonomousAgent

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from networks.brake_network import BrakeNetwork
from agent_utils import base_utils
from agent_utils.pid_controller import PIDController
from agent_utils.planner import RoutePlanner


SENSOR_CONFIG = {
            'width': 400,
            'height': 300,
            'fov': 100
        }


def get_entry_point():
    return 'BrakeAgent'


class BrakeAgent(AutonomousAgent):
    def init_dnn_agent(self):
        input_dims = (3, SENSOR_CONFIG['height'], SENSOR_CONFIG['width'])

        # initialize imitation learning model only
        if self.run_imitation_agent:
            self.device = T.device('cuda') #T.device('cuda' if T.cuda.is_available() else 'cpu')
            print("device: ", self.device)

            # load pretrained policy network
            self.policy = BrakeNetwork(pretrained=False)
            self.policy.to(self.device)

            model_name = "brake_agent_epoch_49.pth"
            trained_policy_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), model_name)
            print(f"trained_policy_path {trained_policy_path}")
            
            self.policy.load_state_dict(T.load(trained_policy_path))
            self.policy.eval()
        else:
            self.device = self.agent.device

    def init_auto_pilot(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_privileged_agent(self):
        self.hero_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.hero_vehicle.get_world()

        self.collision_intensity = 0.0
        self.is_collision = False
        self.is_lane_invasion = False

        if not self.run_imitation_agent:
            self.privileged_sensors()

    def setup(self, rl_model):
        rl_model = None
        if rl_model is None:
            self.run_imitation_agent = True
            self.debug = True
        else:
            self.run_imitation_agent = False

            self.agent = rl_model
            self.debug = self.agent.debug
            self.writer = self.agent.writer

        if not self.run_imitation_agent:
            if not self.agent.evaluate:
                self.best_reward = self.agent.db.get_best_reward(self.agent.training_id)
                self.total_step_num = self.agent.db.get_total_step_num(self.agent.training_id)
                print(f"training model_name {self.agent.model_name}   episode_number {self.agent.db.get_global_episode_number(self.agent.training_id)}   total_step_num {self.total_step_num}   latest_sample_id {self.agent.db.get_latest_sample_id(self.agent.training_id)}   best_reward {self.best_reward}   best_reward_episode_number {self.agent.db.get_best_reward_episode_number(self.agent.training_id)}")
            else:
                self.best_reward = 0.0
                self.total_step_num = self.agent.db.get_evaluation_total_step_num(self.agent.evaluation_id)
                print(f"evaluation model_name {self.agent.model_name}   model_episode_number {self.agent.db.get_evaluation_model_episode_number(self.agent.evaluation_id)}   episode_number {self.agent.db.get_evaluation_global_episode_number(self.agent.evaluation_id)}   total_step_num {self.total_step_num}   average_evaluation_score {self.agent.db.get_evaluation_average_evaluation_score(self.agent.evaluation_id)}")

        self.initialized = False
        self._sensor_data = SENSOR_CONFIG
        
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

        if not self.run_imitation_agent:
            self.collision_sensor = None
            self.lane_invasion_sensor = None

        self.initialized = True

        self.step_number = 1
        self.episode_total_reward = 0.0
        
        # do not initialize if only imitation agent is evaluated
        if not self.run_imitation_agent:
            self.push_buffer = False
            self.next_state = []
            self.count_vehicle_stop = 0
            self.n_updates = 0
            self.total_loss_pi = 0.0
            self.total_loss_q = 0.0

        if self.debug:
            cv2.namedWindow("IL-rgb-front")

    def get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

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
                    }
                ] 
    
    def tick(self, input_data):
        rgb_front = input_data['rgb_front'][1][:, :, :3]
        rgb_front = rgb_front[:, :, ::-1]
        gps = input_data['gps'][1][:2]
        compass = input_data['imu'][1][-1]

        speed = self.hero_vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2) # km/h

        return {
                'rgb_front': rgb_front,
                'gps': gps,
                'speed': speed_kmh,
                'compass': compass
                }

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        data = self.tick(input_data)
        gps = self.get_position(data)
        speed = data['speed']
        compass = data['compass']

        near_node, near_command = self._route_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        # front image
        rgb_image = data['rgb_front']
        cv_image = rgb_image[:, :, ::-1]

        fused_inputs = np.zeros(3, dtype=np.float32)
        fused_inputs[0] = speed / 3.6 # m/s
        fused_inputs[1] = near_node[0] - gps[0]
        fused_inputs[2] = near_node[1] - gps[1]

        if self.debug:
            disp_front_image = cv2.UMat(cv_image)
            cv2.imshow("rgb-front", disp_front_image)
            cv2.waitKey(1)
        
        dnn_agent_brake = self.agent.inference(cv_image, fused_inputs)

        if float(dnn_agent_brake) > 0.5:
            brake = float(1)
        else:
            brake = float(0)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data, brake)

        applied_control = carla.VehicleControl()
        applied_control.throttle = throttle
        applied_control.steer = steer
        applied_control.brake = brake

        return applied_control

    def destroy(self):
        if not self.run_imitation_agent:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
            if self.lane_invasion_sensor is not None:
                self.lane_invasion_sensor.stop()
        
        # terminate and go to another eposide
        os.kill(os.getpid(), signal.SIGINT)

    def _get_control(self, target, far_target, tick_data, brake):
        pos = self.get_position(tick_data)

        theta = tick_data['compass']
        speed = tick_data['speed']

        # steering
        angle_unnorm = base_utils.get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = base_utils.get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4.0 if should_slow else 7.0
        
        if brake > 0.5:
            target_speed = 0.0
        
        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm
        
        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake > 0.5:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed
