from __future__ import print_function

import os
import signal
import sys
import numpy as np
np.random.seed(0)
import carla
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import cv2
import math
import weakref

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import base_utils
from utils.pid_controller import PIDController
from utils.planner import RoutePlanner
from _scenario_runner.srunner.autoagents.autonomous_agent import AutonomousAgent
from networks.offset_network import OffsetNetwork

 # TODO: SENSOR configs can be put to DB

SENSOR_CONFIG = {
            'width': 400,
            'height': 300,
            'fov': 100
        }


class OffsetAgent(AutonomousAgent):

    def init_dnn_agent(self):
        input_dims = (3, SENSOR_CONFIG['height'], SENSOR_CONFIG['width'])

        # initialize imitation learning model only
        if self.run_imitation_agent:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("device: ", self.device)

            # load pretrained policy network
            self.policy = OffsetNetwork()
            self.policy.to(self.device)

            # TODO: IMPORTANT! make a global input parameter
            model_name = "offset_model_epoch_45.pth"
            trained_policy_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), model_name)

            self.policy.load_state_dict(torch.load(trained_policy_path))
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

        self.privileged_sensors()

    def setup(self, rl_model):
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

        self.initialized = True

        # do not initialize if only imitation agent is evaluated
        if not self.run_imitation_agent:
            self.push_buffer = False
            self.next_state = []
            self.step_number = 1
            self.episode_total_reward = 0.0
            self.count_vehicle_stop = 0
            self.n_updates = 0
            self.total_loss_pi = 0.0
            self.total_loss_q = 0.0

        if self.debug:
            cv2.namedWindow("rgb-front")

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
        rgb_front_image = data['rgb_front']
        front_cv_image = rgb_front_image[:, :, ::-1]

        fused_inputs = np.zeros(3, dtype=np.float32)

        fused_inputs[0] = speed / 3.6 # m/s
        fused_inputs[1] = far_node[0] - gps[0]
        fused_inputs[2] = far_node[1] - gps[1]

        if self.debug:
            disp_front_image = cv2.UMat(front_cv_image)
            cv2.imshow("rgb-front", disp_front_image)
            cv2.waitKey(1)

        # construct network input image format
        dnn_input_image = self.image_to_dnn_input(image=front_cv_image)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = torch.from_numpy(fused_inputs.copy()).unsqueeze(0).to(self.device)

        if self.run_imitation_agent:
            with torch.no_grad():
                state_space = self.policy(fronts=dnn_input_image, fused_input=fused_inputs_torch)
            state_space = state_space.cpu().detach().numpy()[0]
            # convert state space to torch tensors to be an input to neural net
            state_torch = torch.from_numpy(state_space.copy()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                dnn_brake, offset = self.policy.compute_action(state_space=state_torch)
            dnn_brake = dnn_brake.cpu().detach().numpy()[0]
            offset = offset.cpu().detach().numpy()[0]

        else:
            # apply freezed pre-trained model onto the image and get state space
            state = self.agent.get_state_space(fronts=dnn_input_image, fused_input=fused_inputs_torch)
            # convert state space to torch tensors to be an input to neural net
            state_torch = torch.from_numpy(state.copy()).unsqueeze(0).to(self.device)

            dnn_brake, offset = self.agent.select_action(network=self.agent.policy, state_space=state_torch)

        agent_action = np.array([dnn_brake[0], offset[0]])

        # left: -, right: +
        new_near_node = self.shift_point(ego_compass=compass, ego_gps=gps, near_node=near_node, offset_amount=offset)

        # get auto-pilot actions
        steer, throttle, target_speed = self.get_control(new_near_node, far_node, data)

        # determine whether to accelerate or brake
        if dnn_brake >= 0.5:
            throttle = 0.0
            brake = 1.0
        else:
            throttle = throttle
            brake = 0.0

        if not self.run_imitation_agent:
            # maintains RL structure
            self.rl_computation(state=state, agent_action=agent_action, dnn_brake=dnn_brake, throttle=throttle, speed=speed, gps=gps, far_node=far_node)

        applied_control = carla.VehicleControl()
        applied_control.throttle = float(throttle)
        applied_control.steer = float(steer)
        applied_control.brake = float(brake)

        print("[Applied Action] :", "throttle:", round(throttle, 2), "steer:", round(steer, 2), "brake:", round(brake, 2), " [Network Output] :", "brake:", round(agent_action[0], 2), "offset:", round(agent_action[1], 2))

        return applied_control

    def rl_computation(self, state, agent_action, dnn_brake, throttle, speed, gps, far_node):
        # compute step reward and deside for termination
        reward, done = self.calculate_reward(dnn_brake=dnn_brake, throttle=throttle, ego_speed=speed, ego_gps=gps, goal_point=far_node)

        policy_loss = None
        value_loss = None

        if self.push_buffer and not self.agent.evaluate:
            # TODO: database format should be changed to be compatible rl transitions
            self.agent.memory.push_transition(state, agent_action, reward, self.next_state, done)

            if self.agent.memory.filled_size > self.agent.batch_size:
                sample_batch = self.agent.memory.sample(self.agent.batch_size)

                policy_loss, value_loss = self.agent.update(sample_batch)
            
                self.n_updates += 1 # number of updates in one episode
                self.total_loss_pi += policy_loss # episodic loss
                self.total_loss_q += value_loss # episodic loss

        self.next_state = state
        
        self.push_buffer = True

        self.episode_total_reward += reward

        # terminate an episode
        if done:
            if not self.agent.evaluate: # training
                self.agent.db.update_latest_sample_id(self.agent.memory.id, self.agent.training_id)
                self.agent.db.update_total_step_num(self.total_step_num, self.agent.training_id)

                if self.episode_total_reward > self.best_reward:
                    self.best_reward = self.episode_total_reward
                    self.agent.db.update_best_reward(self.best_reward, self.agent.training_id)
                    self.agent.db.update_best_reward_episode_number(self.agent.db.get_global_episode_number(self.agent.training_id), self.agent.training_id)

                    print("Best Episode Reward: ", self.best_reward)

                    self.agent.save_models(self.agent.db.get_global_episode_number(self.agent.training_id))

                base_utils.tensorboard_writer(self.writer, self.agent.db.get_global_episode_number(self.agent.training_id), self.episode_total_reward, self.best_reward, self.total_loss_pi, self.total_loss_q, self.n_updates)
            
            else: # evaluation
                self.agent.db.update_evaluation_total_step_num(self.total_step_num, self.agent.evaluation_id)
                self.agent.db.update_evaluation_average_evaluation_score(self.episode_total_reward, self.agent.evaluation_id)
                base_utils.tensorboard_writer_evaluation(self.writer, self.agent.db.get_evaluation_global_episode_number(self.agent.evaluation_id), self.episode_total_reward)

            print("------------ Terminating! ------------")
            print("Episode Total Reward: ", round(self.episode_total_reward, 3))
            self.destroy()

        self.step_number += 1
        self.total_step_num += 1

    # TODO: if you change the reward, save the snippet and save the id of it to DB
    def calculate_reward(self, dnn_brake, throttle, ego_speed, ego_gps, goal_point):
        reward = -0.1
        done = 0

        reward += ego_speed

        # distance to each far distance goal points in meters
        distance = np.linalg.norm(goal_point - ego_gps)

        # if any of the following is not None, then the agent should brake
        is_light, is_walker, is_vehicle = self.traffic_data()

        # give penalty if ego vehicle is not braking where it should brake
        if any(x is not None for x in [is_light, is_walker, is_vehicle]):
            print("[Scenario]: traffic light-", is_light, " walker-", is_walker, " vehicle-", is_vehicle)
            
            # accelerating while it should brake
            if dnn_brake < 0.2:
                print("[Penalty]: not braking !")
                reward -= ego_speed * throttle
            
            # braking as it should be
            else:
                print("[Reward]: correctly braking !")
                reward += 15

        else:
            if ego_speed <= 0.5:
                self.count_vehicle_stop += 1
            else:
                self.count_vehicle_stop = 0

            # terminate if vehicle is not moving for too long steps
            if self.count_vehicle_stop > 100:
                print("[Penalty]: too long stopping !")
                reward -= 20
                done = 1

        # negative reward for collision or lane invasion
        if self.is_lane_invasion:
            print("[Penalty]: lane invasion !")
            reward -= 50
            
        if self.is_collision:
            print("[Penalty]: collision !")
            reward -= 100 * self.collision_intensity
            done = 1

        return reward, done

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

    def privileged_sensors(self):
        blueprint = self.world.get_blueprint_library()

        # get blueprint of the sensors
        bp_collision = blueprint.find('sensor.other.collision')
        bp_lane_invasion = blueprint.find('sensor.other.lane_invasion')

        # attach sensors to the ego vehicle
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)
        self.lane_invasion_sensor = self.world.spawn_actor(bp_lane_invasion, carla.Transform(), attach_to=self.hero_vehicle)

        # create sensor event callbacks
        self.collision_sensor.listen(lambda event: OffsetAgent._on_collision(weakref.ref(self), event))
        self.lane_invasion_sensor.listen(lambda event: OffsetAgent._on_lane_invasion(weakref.ref(self), event))

    def traffic_data(self):
        all_actors = self.world.get_actors()

        lights_list = all_actors.filter('*traffic_light*')
        walkers_list = all_actors.filter('*walker*')
        vehicle_list = all_actors.filter('*vehicle*')

        traffic_lights = base_utils.get_nearby_lights(self.hero_vehicle, lights_list)

        light = self.is_light_red(traffic_lights)
        walker = self.is_walker_hazard(walkers_list)
        vehicle = self.is_vehicle_hazard(vehicle_list)

        return light, walker, vehicle

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

        return steer, throttle, target_speed

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

    def is_light_red(self, traffic_lights):
        if self.hero_vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self.hero_vehicle.get_traffic_light()
            for light in traffic_lights:
                if light.id == affecting.id:
                    return affecting
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

    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        self.is_lane_invasion = True  

    def destroy(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.stop()

        # terminate and go to another eposide
        os.kill(os.getpid(), signal.SIGINT)
