from __future__ import print_function

import os
import signal
import sys
import numpy as np
import carla
import torch
import cv2

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.pid_controller import PIDController
from utils.planner import RoutePlanner
from _scenario_runner.srunner.autoagents.autonomous_agent import AutonomousAgent


SENSOR_CONFIG = {
            'width': 400,
            'height': 300,
            'fov': 60
        }


class WaypointAgent(AutonomousAgent):

    def init_dnn_agent(self):
        input_dims = (3, SENSOR_CONFIG['height'], SENSOR_CONFIG['width'])
        n_actions = 2

        print(f"input_dims: {input_dims}\nn_actions: {n_actions}")

        checkpoint_dir = parent + os.path.sep + "models"
        print(f"models will be saved to {checkpoint_dir}")

        self.device = self.agent.device

    def init_auto_pilot(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def setup(self, rl_model):
        self.agent = rl_model

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

        self.init_auto_pilot()
        self.init_dnn_agent()

        self.initialized = True
        self.push_buffer = False

        self.next_image_features = []
        self.next_fused_inputs = []

        cv2.namedWindow("rgb-front-FOV-60")

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
                    },
                # { 
                #     'type': 'sensor.speedometer',
                #     'reading_frequency': 20,
                #     'id': 'speed'
                #     }
                ]   
    
    def tick(self, input_data):
        rgb_front = input_data['rgb_front'][1][:, :, :3]
        rgb_front = rgb_front[:, :, ::-1]
        gps = input_data['gps'][1][:2]
        # speed = input_data['speed'][1]['speed'] # TODO: open speed sensor
        speed = 20.0
        compass = input_data['imu'][1][-1]

        return {
                'rgb_front': rgb_front,
                'gps': gps,
                'speed': speed,
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
        rgb_front_image = data['rgb_front']
        front_cv_image = rgb_front_image[:, :, ::-1]

        fused_inputs = np.zeros(3, dtype=np.float32)

        fused_inputs[0] = speed
        fused_inputs[1] = near_node[0] - gps[0]
        fused_inputs[2] = near_node[1] - gps[1]

        disp_front_image = cv2.UMat(front_cv_image)
        cv2.imshow("rgb-front-FOV-60", disp_front_image)
        cv2.waitKey(1)

        # construct network input image format
        dnn_input_image = self.image_to_dnn_input(image=front_cv_image)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = torch.from_numpy(fused_inputs.copy()).unsqueeze(0).to(self.device)

        # apply freezed pre-trained resnet model onto the image
        image_features_torch = self.agent.resnet_backbone(dnn_input_image)
        image_features = image_features_torch.cpu().detach().numpy()[0]

        # get action from actor network
        dnn_agent_action = np.array(self.agent.select_action(image_features=image_features_torch, fused_input=fused_inputs_torch))

        # TODO: left here, compute reward and do not save buffer in a ram
        reward = 0
        done = 1
        batch_size = 64

        if self.push_buffer:
            self.agent.memory.push(image_features, fused_inputs, dnn_agent_action, reward, self.next_image_features, self.next_fused_inputs, done)

            if len(self.agent.memory.memories) > batch_size:
                self.agent.update(self.agent.memory.sample(batch_size))

        self.next_image_features = image_features
        self.next_fused_inputs = fused_inputs

        # determine whether to accelerate or brake
        if float(dnn_agent_action[1]) >= 0.0:
            brake = 0.0
            throttle = dnn_agent_action[1]
            
            if throttle >= 0.75:
                throttle = 0.75
        else:
            brake = 1.0
            throttle = 0.0

        steer = float(dnn_agent_action[0])

        # TODO: removed PID controller's actions
        # steer, throttle, brake, target_speed = self.get_control(near_node, far_node, data, brake)

        print("(throttle, steer, brake) : ", throttle, steer, brake)

        applied_control = carla.VehicleControl()
        applied_control.throttle = float(throttle)
        applied_control.steer = steer
        applied_control.brake = brake
        
        self.push_buffer = True

        return applied_control

    def get_control(self, target, far_target, tick_data, brake):
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
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        
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

    def destroy(self):
        # terminate and go to another eposide
        os.kill(os.getpid(), signal.SIGINT)
