from __future__ import print_function

import os
import sys
import time
import numpy as np
np.random.seed(0)
import cv2
import math
import torch
torch.manual_seed(0)
import carla
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from networks.brake_network import BrakeNetwork
from utils import base_utils
from utils.pid_controller import PIDController
from utils.planner import RoutePlanner
from _scenario_runner.srunner.autoagents import autonomous_agent


def get_entry_point():
    return 'BrakeAgent'


class BrakeAgent(autonomous_agent.AutonomousAgent):

    def setup(self, rl_model):
        self.rl_model = rl_model
        self.debug = self.rl_model.debug
        self.writer = self.rl_model.writer

        if not self.rl_model.evaluate:
            self.best_reward = self.rl_model.db.get_best_reward(self.rl_model.training_id)
            self.total_step_num = self.rl_model.db.get_total_step_num(self.rl_model.training_id)
            print(f"training model_name {self.rl_model.model_name}   episode_number {self.rl_model.db.get_global_episode_number(self.rl_model.training_id)}   total_step_num {self.total_step_num}   latest_sample_id {self.rl_model.db.get_latest_sample_id(self.rl_model.training_id)}   best_reward {self.best_reward}   best_reward_episode_number {self.rl_model.db.get_best_reward_episode_number(self.rl_model.training_id)}")
        else:
            self.best_reward = 0.0
            self.total_step_num = self.rl_model.db.get_evaluation_total_step_num(self.rl_model.evaluation_id)
            print(f"evaluation model_name {self.rl_model.model_name}   model_episode_number {self.rl_model.db.get_evaluation_model_episode_number(self.rl_model.evaluation_id)}   episode_number {self.rl_model.db.get_evaluation_global_episode_number(self.agent.evaluation_id)}   total_step_num {self.total_step_num}   average_evaluation_score {self.rl_model.db.get_evaluation_average_evaluation_score(self.rl_model.evaluation_id)}")

        self.device = self.rl_model.device
        self.track = autonomous_agent.Track.SENSORS
        
        self.wall_start = time.time()
        self.initialized = False

        self._sensor_data = {
            'width': 400,
            'height': 300,
            'fov': 100
        }

        model_name = "brake_agent_epoch_49.pth"
        self.model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), model_name)

        # init agent (do not load resnet trained weights as model will be completely loaded)
        self.agent = BrakeNetwork(pretrained=False)
        self.agent.load_state_dict(torch.load(self.model_path))

        self.agent.to(self.device)
        self.agent.eval()

        self.hero_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.hero_vehicle.get_world()
        
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner.set_route(self._global_plan, True)

        self._init_auto_pilot()

        self.initialized = True

        self.next_state = []

        self.step_number = 1
        self.episode_total_reward = 0.0
        self.count_vehicle_stop = 0
        self.n_updates = 0
        self.total_loss_pi = 0.0
        self.total_loss_q = 0.0

        if self.debug:
            cv2.namedWindow("rgb-front")

    def _get_position(self, tick_data):
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
        gps = self._get_position(data)
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
        
        # TODO : left here (define state space for RL) (put self.agent inside dqn.py and freeze network in dqn file)
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
        del self.agent

    def _init_auto_pilot(self):
        # pid controllers of auto_pilot
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def _get_control(self, target, far_target, tick_data, brake):
        pos = self._get_position(tick_data)

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
