import os
import sys
import weakref
import numpy as np
import random
import cv2
import json
import carla
import torch

from torchvision import models

from datetime import datetime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent 

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from collections import deque
from agent_utils import base_utils
from agent_utils.pid_controller import PIDController
from agent_utils.planner import RoutePlanner

from networks.imitation_network import ImitationNetwork
from networks.dqn_network import DQNNetwork
from networks.policy_classifier_network import PolicyClassifierNetwork


# if True: IL vs Autopilot will be checked and Autopilot data is applied when brake commands are not the same
# if False: IL agent will always be applied without saving data
DAGGER = False

# if True: front camera will be displayed
DEBUG = False

# if True: policy classifier will be activated
ACTIVATE_SWITCH = True

# if False: only clear weather will be active
IS_WEATHER_CHANGE = True

# trained IL model path (should be inside "checkpoint/models/" folder)
model_folder = "Feb_05_2022-15_18_48_dagger_2_big" #"Feb_04_2022-19_14_19_imitation_0"
model_name = "epoch_15.pth"

# stuck vehicle RL model
rl_model_folder = "Feb_07_2022-14_26_30"
rl_model_eps_num = 300

# policy switch classifier model
policy_model_folder = "Feb_05_2022-21_29_35_policy_classifier"
switch_model_name = "epoch_30.pth"


SENSOR_CONFIG = {
    'width': 400,
    'height': 300,
    'high_fov': 100,
    'low_fov': 60
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
    return 'ImitationAgent'


class ImitationAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_id, rl_model=None):
        self.track = autonomous_agent.Track.SENSORS
        self._sensor_data = SENSOR_CONFIG
        self.config_path = path_to_conf_file
        self.debug = DEBUG
        self.step = -1
        self.stop_step = 0
        self.not_brake_step = 0
        self.data_count = 0
        self.initialized = False
        self.route_id = route_id
        self.weather_id = WEATHERS_IDS[0]

        today = datetime.today() # month - date - year
        now = datetime.now() # hours - minutes - seconds

        current_date = str(today.strftime("%b_%d_%Y"))
        current_time = str(now.strftime("%H_%M_%S"))

        # month_date_year-hour_minute_second
        time_info = "/" + current_date + "-" + current_time + "/"

        self.dataset_save_path = os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/dataset/" + os.environ.get('SAVE_DATASET_NAME') + time_info)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device: ", self.device)

        model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/" + model_folder), model_name)
        self.agent = ImitationNetwork(device=self.device)
        self.agent.to(self.device)
        self.agent.load_state_dict(torch.load(model_path))
        for param in self.agent.parameters():
            param.requires_grad = False
        self.agent.eval()

        if ACTIVATE_SWITCH:
            # model selection switcher network 6 modes
            policy_model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/" + policy_model_folder), switch_model_name)
            self.policy_classifier = PolicyClassifierNetwork(device=self.device)
            self.policy_classifier.to(self.device)
            self.policy_classifier.load_state_dict(torch.load(policy_model_path))
            for param in self.policy_classifier.parameters():
                param.requires_grad = False
            self.policy_classifier.eval()

            # load pretrained ResNet and freeze weights
            resnet_model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), "resnet50.zip")
            self.resnet50 = models.resnet50(pretrained=False)
            self.resnet50.to(self.device)
            self.resnet50.load_state_dict(torch.load(resnet_model_path))
            for param in self.resnet50.parameters():
                param.requires_grad = False
            self.resnet50.eval()

            rl_model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/" + rl_model_folder), "dqn" + "-ep_" + str(rl_model_eps_num))
            self.rl_agent = DQNNetwork(state_size=1000, n_actions=4, device=self.device)
            self.rl_agent.to(self.device)
            self.rl_agent.load_state_dict(torch.load(rl_model_path))
            for param in self.rl_agent.parameters():
                param.requires_grad = False
            self.rl_agent.eval()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)

        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner.set_route(self._global_plan, True)

        if DAGGER:
            self.init_dataset(output_dir=self.dataset_save_path)
        self.init_auto_pilot()
        self.init_privileged_agent()

        self.initialized = True

        self.count_vehicle_stop = 0
        self.count_is_seen = 0

        self.switch2rl = False
        self.count_rl_acts = 0
        self.max_rl_act_steps = 100
        
        self.policy_label = 0
        self.check_count = 0

        self.previous_gps = 0.0
        self.speed_sequence = deque(np.zeros(120), maxlen=120)

        if self.debug:
            cv2.namedWindow("rgb-front")
    
    def init_dataset(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.subfolder_paths = []
        subfolders = ["rgb_front", "rgb_front_60", "rgb_rear", "measurements"]

        for subfolder in subfolders:
            self.subfolder_paths.append(os.path.join(output_dir, subfolder))
            if not os.path.exists(self.subfolder_paths[-1]):
                os.makedirs(self.subfolder_paths[-1])

    def init_auto_pilot(self):
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def init_privileged_agent(self):
        self.hero_vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.hero_vehicle.get_world()

        self.privileged_sensors()

    def privileged_sensors(self):
        blueprint = self.world.get_blueprint_library()

        # get blueprint of the sensor
        bp_collision = blueprint.find('sensor.other.collision')

        self.is_collision = False

        # attach collision sensor to the ego vehicle
        self.collision_sensor = self.world.spawn_actor(bp_collision, carla.Transform(), attach_to=self.hero_vehicle)

        # create sensor event callbacks
        self.collision_sensor.listen(lambda event: ImitationAgent._on_collision(weakref.ref(self), event))

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
                    'id': 'rgb_front'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['low_fov'],
                    'id': 'rgb_front_60'
                    },
                {
					'type': 'sensor.camera.rgb',
					'x': -1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
					'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['high_fov'],
					'id': 'rgb_rear'
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

    def stop_function(self, is_stop):
        if self.stop_step < 200 and is_stop is not None:
            self.stop_step += 1
            self.not_brake_step = 0
            return True
       
        else:
            if self.not_brake_step < 300:
                self.not_brake_step += 1 
            else:
                self.stop_step = 0
            return None

    def check_daggerity(self, autopilot_brake, model_brake):
        if abs(autopilot_brake - model_brake) > 0.5:
            return True
        else:
            return False

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        data = self.tick(input_data)
        gps = self.get_position(data)
        speed = data['speed']
        compass = data['compass']

        if self.step % 20 == 0 and IS_WEATHER_CHANGE:
            self.change_weather()

        if self.step % 10 == 0:
            self.speed_sequence.append(speed)
            self.previous_gps = gps

        # fix for theta=nan in some measurements
        if np.isnan(data['compass']):
            ego_theta = 0.0
        else:
            ego_theta = compass

        near_node, near_command = self._route_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        # front image
        rgb_front_image = data['rgb_front']
        front_cv_image = rgb_front_image[:, :, ::-1]

        rgb_front_60_image = data['rgb_front_60']
        front_60_cv_image = rgb_front_60_image[:, :, ::-1]

        rgb_rear_image = data['rgb_rear']
        rear_cv_image = rgb_rear_image[:, :, ::-1]

        fused_inputs = np.zeros(2, dtype=np.float32)
        fused_inputs[0] = near_node[0] - gps[0]
        fused_inputs[1] = near_node[1] - gps[1]

        if self.debug:
            disp_front_image = cv2.UMat(front_cv_image)
            cv2.imshow("rgb-front", disp_front_image)
            cv2.waitKey(1)

        # run brake classifier imitation learning agent
        dnn_agent_brake = self.agent.inference(front_images=front_60_cv_image, waypoint_input=fused_inputs, speed_sequence=np.array(self.speed_sequence))

        # when initialized, wait 50 steps until vehicle starts to move
        if ACTIVATE_SWITCH and self.step > 50:
            with torch.no_grad():
                self.policy_label = self.policy_classifier.inference(front_images=front_60_cv_image, speed_sequence=np.array(self.speed_sequence))

            # TODO: train different DQN agents
            # 0 class corresponds to running an imitation learning model
            if self.policy_label == 3 and self.check_count > 3:
                self.switch2rl = True
                self.check_count = 0
            else:
                self.check_count += 1

        if self.switch2rl:
            dnn_input_image = self.rl_agent.image_to_dnn_input(image=front_cv_image)

            # apply freezed pre-trained resnet model onto the image
            with torch.no_grad():
                image_features_torch = self.resnet50(dnn_input_image)
                
                # TODO: add actions of another RL models here (right now all models are stuck vehicle case)
                # if self.policy_label != 0:
                if self.count_rl_acts < self.max_rl_act_steps:
                    self.count_rl_acts += 1

                    print("[Info]: Stuck Vehicle RL is Running ! (class id -> 3)")
                    action_value = self.rl_agent(image_features_torch)

                    max_value = torch.argmax(action_value).unsqueeze(0).unsqueeze(0).cpu().detach().numpy()[0][0]
                
                else:
                    pass

            # apply rl action constant number of times
            if self.count_rl_acts < self.max_rl_act_steps:
                # update near node by shifting future nodes to 90 degrees left or right
                dnn_agent_brake, near_node = self.calculate_high_level_action(high_level_action=max_value, compass=ego_theta, gps=gps, near_node=near_node)
            
            else:
                self.switch2rl = False
                self.count_rl_acts = 0
            
        # if any of the following is not None, then the agent should brake
        is_light, is_walker, is_vehicle, is_stop = self.traffic_data()
        print("[Traffic]: traffic light-", is_light, " walker-", is_walker, " vehicle-", is_vehicle, " stop-", is_stop)

        # by using priviledged information determine braking
        is_brake = any(x is not None for x in [is_light, is_walker, is_vehicle, is_stop])

        if self.check_daggerity(autopilot_brake=is_brake, model_brake=int(dnn_agent_brake)) and DAGGER:
            print("[Info]: DAgger is Active !")

            save_dagger = True
            applied_brake = is_brake
        
        else:
            save_dagger = False
            applied_brake = dnn_agent_brake

        # apply pid controllers
        steer, throttle, brake, target_speed, angle = self.get_control(target=near_node, far_target=far_node, tick_data=data, brake=applied_brake)

        # compute step reward and deside for termination
        reward, done = self.calculate_reward(throttle=throttle, ego_speed=speed, ego_gps=gps, is_light=is_light, is_vehicle=is_vehicle, is_walker=is_walker, is_stop=is_stop)
        label = self.define_classifier_label(reward)

        print("[Action]:", throttle, steer, brake, " [Reward]:", reward, " [Done]:", done, "[Waypoint]:", near_node, "[GT Class ID]:", label, "[Predicted Class ID]:", self.policy_label)

        self.is_collision = False

        applied_control = carla.VehicleControl()
        applied_control.throttle = throttle
        applied_control.steer = steer
        applied_control.brake = brake

        measurement_data = {
            'x': gps[0],
            'y': gps[1],

            'speed': speed,
            'theta': ego_theta,

            'x_command': far_node[0],
            'y_command': far_node[1],
            'far_command': far_command.value,

            'near_node_x': near_node[0],
            'near_node_y': near_node[1],
            'near_command': near_command.value,

            'steer': applied_control.steer,
            'throttle': applied_control.throttle,
            'brake': applied_control.brake,

            'should_slow': self.should_slow,
            'should_brake': self.should_brake,

            'angle': self.angle,
            'angle_unnorm': self.angle_unnorm,
            'angle_far_unnorm': self.angle_far_unnorm,

            'is_vehicle_present': self.is_vehicle_present,
            'is_pedestrian_present': self.is_pedestrian_present,
            'is_red_light_present': self.is_red_light_present,
            'is_stops_present': self.is_stops_present,

            'weather_id': self.weather_id,

            'reward': reward,
            'done': done,

            'speed_sequence': np.array(self.speed_sequence).tolist(),
            'label': label
            }

        if self.step % 10 == 0 and save_dagger:
            self.save_data(image_front=front_cv_image, image_front_60=front_60_cv_image, image_rear=rear_cv_image, data=measurement_data)
        
        return applied_control

    def tick(self, input_data):
        self.step += 1

        rgb_front = input_data['rgb_front'][1][:, :, :3]
        rgb_front = rgb_front[:, :, ::-1]
        
        rgb_front_60 = input_data['rgb_front_60'][1][:, :, :3]
        rgb_front_60 = rgb_front_60[:, :, ::-1]
        
        rgb_rear = input_data['rgb_rear'][1][:, :, :3]
        rgb_rear = rgb_rear[:, :, ::-1]
        
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        return {
            'rgb_front': rgb_front,
            'rgb_front_60': rgb_front_60,
            'rgb_rear': rgb_rear,
            'gps': gps,
            'speed': speed,
            'compass': compass
            }

    def get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps
    
    def get_control(self, target, far_target, tick_data, brake):
        pos = self.get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # steering
        angle_unnorm = base_utils.get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # acceleration
        angle_far_unnorm = base_utils.get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4.0 if should_slow else 7.0
        
        if brake:
            target_speed = 0.0
        
        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm
        
        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if self.should_brake > 0.5:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed, angle

    def destroy(self):
        del self.agent

        if ACTIVATE_SWITCH:
            del self.rl_agent
            del self.policy_classifier
            del self.resnet50

        if self.collision_sensor is not None:
            self.collision_sensor.stop()

    def define_classifier_label(self, reward):
        label = 0
        if reward < 0.0:
            if self.is_red_light_present:
                label = 1
            elif self.is_stops_present:
                label = 2
            elif self.is_vehicle_present:
                label = 3
            elif self.is_pedestrian_present:
                label = 4
            else:
                label = 5
        else:
            label = 0
        return label

    def calculate_high_level_action(self, high_level_action, compass, gps, near_node):
        # 0 -> brake
        # 1 -> go to left lane of next_waypoint
        # 2 -> keep lane
        # 3 -> go to right lane of next_waypoint

        if high_level_action == 1: # left
            offset = -3.5
            new_near_node = self.shift_point(ego_compass=compass, ego_gps=gps, near_node=near_node, offset_amount=offset)
        elif high_level_action == 3: # right
            offset = 3.5
            new_near_node = self.shift_point(ego_compass=compass, ego_gps=gps, near_node=near_node, offset_amount=offset)
        else: # keep lane
            offset = 0.0
            new_near_node = near_node

        if high_level_action == 0:
            brake = 1.0
        else:
            brake = 0.0

        return brake, new_near_node

    def traffic_data(self):
        all_actors = self.world.get_actors()

        lights_list = all_actors.filter('*traffic_light*')
        walkers_list = all_actors.filter('*walker*')
        vehicle_list = all_actors.filter('*vehicle*')
        stop_list = all_actors.filter('*stop*')

        traffic_lights = base_utils.get_nearby_lights(self.hero_vehicle, lights_list)
        stops = base_utils.get_nearby_lights(self.hero_vehicle, stop_list)

        if len(stops) == 0:
            stop = None
        else:
            stop = stops

        light = self.is_light_red(traffic_lights)
        walker = self.is_walker_hazard(walkers_list)
        vehicle = self.is_vehicle_hazard(vehicle_list)
        stop = self.stop_function(stop)

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_red_light_present = 1 if light is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        self.is_stops_present = 1 if stop is not None else 0

        return light, walker, vehicle, stop

    def calculate_reward(self, throttle, ego_speed, ego_gps, is_light, is_vehicle, is_walker, is_stop):
        reward = 0.0
        done = 0

        # give penalty if ego vehicle is not braking where it should brake
        if any(x is not None for x in [is_light, is_walker, is_vehicle]):
            self.count_is_seen += 1
            
            # throttle desired after too much waiting around vehicle or walker
            if self.count_is_seen > 1200:
                print("[Penalty]: too much stopping when there is a vehicle or walker or stop sign or traffic light around !")
                reward -= 100

            # braking desired
            else:
                # accelerating while it should brake
                if throttle < 0.1:
                    print("[Reward]: correctly braking !")
                    reward += 50
                else:
                    print("[Penalty]: not braking !")
                reward -= ego_speed

        else:
            reward += ego_speed
            self.count_is_seen = 0
        
        # distance from starting position
        reward += np.linalg.norm(ego_gps - self.previous_gps)

        # negative reward for collision
        if self.is_collision:
            print("[Penalty]: collision !")
            reward -= 1500
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

    def is_light_red(self, traffic_lights):
        for light in traffic_lights:
            if light.get_state() == carla.TrafficLightState.Red:
                return True
            elif light.get_state() == carla.TrafficLightState.Yellow:
                return True
        return None

    def is_walker_hazard(self, walkers_list):
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        v1 = 10.0 * base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        for walker in walkers_list:
            v2_hat = base_utils._orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(base_utils._numpy(walker.get_velocity()))
            if s2 < 0.05:
                v2_hat *= s2
            p2 = -3.0 * v2_hat + base_utils._numpy(walker.get_location())
            v2 = 8.0 * v2_hat
            collides, collision_point = base_utils.get_collision(p1, v1, p2, v2)
            if collides:
                return walker
        return None

    def is_vehicle_hazard(self, vehicle_list):
        o1 = base_utils._orientation(self.hero_vehicle.get_transform().rotation.yaw)
        p1 = base_utils._numpy(self.hero_vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(base_utils._numpy(self.hero_vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self.hero_vehicle.id:
                continue
            o2 = base_utils._orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = base_utils._numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(base_utils._numpy(target_vehicle.get_velocity())))
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

    def save_data(self, image_front, image_front_60, image_rear, data):
        cv2.imwrite(os.path.join(self.subfolder_paths[0], "%04i.png" % self.data_count), image_front)
        cv2.imwrite(os.path.join(self.subfolder_paths[1], "%04i.png" % self.data_count), image_front_60)
        cv2.imwrite(os.path.join(self.subfolder_paths[2], "%04i.png" % self.data_count), image_rear)

        with open(os.path.join(self.subfolder_paths[3], "%04i.json" % self.data_count), 'w+', encoding='utf-8') as f:
            json.dump(data, f,  ensure_ascii=False, indent=4)
        
        self.data_count += 1
    
    def change_weather(self):
        index = random.choice(range(len(WEATHERS)))
        self.weather_id = WEATHERS_IDS[index]
        self.world.set_weather(WEATHERS[WEATHERS_IDS[index]])
    
    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.is_collision = True