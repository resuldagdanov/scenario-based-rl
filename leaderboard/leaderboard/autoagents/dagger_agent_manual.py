import os
import json
import datetime
import pathlib
import time
import cv2
import carla
import random
import torch
import carla
import math
import numpy as np

seed = 0
np.random.seed(0)
random.seed(seed) 

from PIL import Image
from collections import deque
from leaderboard.autoagents import autonomous_agent
from transfuser.model import TransFuser
from transfuser.config import GlobalConfig
from transfuser.data import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from transfuser.planner import RoutePlanner
from transfuser.pid_controller import PIDController
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from matplotlib import cm
from tensorboardX import SummaryWriter


SAVE_PATH = os.environ.get('SAVE_PATH', None)

DAGGER_ACTION_THRESHOLD = 0.3
DAGGER_WAYPOINT_THRESHOLD = 3 # meters
DATASET_LIMIT = 10000  # 10K dagger steps for one iteration of imitation training with dagger
BRAKE_LIMIT = 100  # number of steps the autopilot gives brake action

dataset_count = 0
dagger_index = 0
total_steps = -1

SAVE_DATA = False  # saves autopilot and transfuser action data

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


def weather_to_dict(carla_weather):
	weather = {
		'cloudiness': carla_weather.cloudiness,
		'precipitation': carla_weather.precipitation,
		'precipitation_deposits': carla_weather.precipitation_deposits,
		'wind_intensity': carla_weather.wind_intensity,
		'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
		'sun_altitude_angle': carla_weather.sun_altitude_angle,
		'fog_density': carla_weather.fog_density,
		'fog_distance': carla_weather.fog_distance,
		'wetness': carla_weather.wetness,
		'fog_falloff': carla_weather.fog_falloff,
	}
	return weather


def to_orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def to_numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def get_point_inside_boundingbox(point, bb_center, bb_extent):
	A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
	B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
	D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
	M = carla.Vector2D(point.x, point.y)

	AB = B - A
	AD = D - A
	AM = M - A
	am_ab = AM.x * AB.x + AM.y * AB.y
	ab_ab = AB.x * AB.x + AB.y * AB.y
	am_ad = AM.x * AD.x + AM.y * AD.y
	ad_ad = AD.x * AD.x + AD.y * AD.y

	return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad


def get_angle_to(pos, theta, target):
	R = np.array([
		[np.cos(theta), -np.sin(theta)],
		[np.sin(theta),  np.cos(theta)],
		])
		
	aim = R.T.dot(target - pos)
	angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
	angle = 0.0 if np.isnan(angle) else angle 
	
	return angle


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        result.append(light)

    return result


def get_entry_point():
	return 'TransFuserAgent'


def dagger_metric_waypoint(meta_autopilot, meta_transfuser):
	near_node_x_error = meta_autopilot['near_node_x'] - meta_transfuser['near_node'][0]
	near_node_y_error = meta_autopilot['near_node_y'] - meta_transfuser['near_node'][1]

	error_near = np.sqrt(near_node_x_error**2 + near_node_y_error**2)

	if error_near > DAGGER_WAYPOINT_THRESHOLD:
		return True, error_near
	else:
		return False, error_near


def dagger_metric_action(autopilot_actions, network_actions):
	throttle_expert = autopilot_actions.throttle
	steer_expert = autopilot_actions.steer
	brake_expert = autopilot_actions.brake

	throttle_agent = network_actions.throttle
	steer_agent = network_actions.steer
	brake_agent = network_actions.brake

	# shift steering output
	steer_expert = steer_expert + 1
	steer_agent = steer_agent + 1

	if abs(throttle_agent - throttle_expert) > DAGGER_ACTION_THRESHOLD or \
		abs(steer_agent - steer_expert) > DAGGER_ACTION_THRESHOLD or \
		abs(brake_agent - brake_expert) > DAGGER_ACTION_THRESHOLD:
		return True
	else:
		return False


class TransFuserAgent(autonomous_agent.AutonomousAgent):
	
	PROXIMITY_THRESHOLD = 30.0  # meters
	SPEED_THRESHOLD = 0.1 # for stop signs
	WAYPOINT_STEP = 1.0  # meters

	def setup(self, path_to_conf_file):
		self.lidar_processed = list()

		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file

		self.step = -1
		self.brake_count = 0  # check how many steps an autopilot gives brake action

		self.wall_start = time.time()
		self.initialized = False

		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		self.config = GlobalConfig()
		self.net = TransFuser(self.config, 'cuda')
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		self.save_path = None

		if SAVE_PATH is not None:
			now = datetime.datetime.now()

			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			
			self.dataset_name = "D" + str(dataset_count)

			self.dagger_path = pathlib.Path(os.environ['SAVE_PATH']) / "dagger" / string
			(self.dagger_path / self.dataset_name / "log").mkdir(parents=True, exist_ok=True)
			
			self.log_writer = SummaryWriter(logdir=self.dagger_path / self.dataset_name / "log", comment="_" + string)

			(self.dagger_path / self.dataset_name / 'rgb').mkdir(parents=True, exist_ok=True)
			(self.dagger_path / self.dataset_name / 'rgb_left').mkdir(parents=True, exist_ok=True)
			(self.dagger_path / self.dataset_name / 'rgb_right').mkdir(parents=True, exist_ok=True)
			(self.dagger_path / self.dataset_name / 'rgb_rear').mkdir(parents=True, exist_ok=True)
			(self.dagger_path / self.dataset_name / 'lidar').mkdir(parents=True, exist_ok=True)
			(self.dagger_path / self.dataset_name / 'measurements').mkdir(parents=True, exist_ok=True)

			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print(string)

			if SAVE_DATA:
				self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
				self.save_path.mkdir(parents=True, exist_ok=False)

				(self.save_path / 'transfuser' / 'rgb').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'transfuser' / 'lidar_0').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'transfuser' / 'lidar_1').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'transfuser' / 'meta').mkdir(parents=True, exist_ok=False)

				(self.save_path / 'autopilot' / 'rgb').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'autopilot' / 'rgb_left').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'autopilot' / 'rgb_right').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'autopilot' / 'rgb_rear').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'autopilot' / 'lidar').mkdir(parents=True, exist_ok=False)
				(self.save_path / 'autopilot' / 'measurements').mkdir(parents=True, exist_ok=False)

	def init(self):
		self._route_planner = RoutePlanner(min_distance=4.0, max_distance=50.0)
		self._route_planner.set_route(self._global_plan, True)

		self._waypoint_planner = RoutePlanner(min_distance=4.0, max_distance=50.0)
		self._waypoint_planner.set_route(self._plan_gps_HACK, True)

		self._command_planner = RoutePlanner(min_distance=4.0, max_distance=50.0, debug_size=257)
		self._command_planner.set_route(self._global_plan, True)

		self._vehicle = CarlaDataProvider.get_hero_actor()
		self._world = self._vehicle.get_world()

		self.initialized = True

		self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
		self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

		# for stop signs
		self._target_stop_sign = None # the stop sign affecting the ego vehicle
		self._stop_completed = False # if the ego vehicle has completed the stop sign
		self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

		self._traffic_lights = list()

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_left'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_right'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_rear'
					},
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar'
                    },
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					}
				]

	def get_auto_pilot_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._command_planner.mean) * self._command_planner.scale
		return gps

	def get_transfuser_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale
		return gps
		
	def get_auto_pilot_control(self, target, far_target, tick_data):
		pos = self.get_auto_pilot_position(tick_data)
		theta = tick_data['compass']
		speed = tick_data['speed']

		# Steering.
		angle_unnorm = get_angle_to(pos, theta, target)
		angle = angle_unnorm / 90

		steer = self._turn_controller.step(angle)
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)

		# Acceleration.
		angle_far_unnorm = get_angle_to(pos, theta, far_target)
		should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
		target_speed = 4.0 if should_slow else 7.0
		brake = self._should_brake()
		target_speed = target_speed if not brake else 0.0
		
		self.should_slow = int(should_slow)
		self.should_brake = int(brake)
		self.angle = angle
		self.angle_unnorm = angle_unnorm
		self.angle_far_unnorm = angle_far_unnorm
		
		delta = np.clip(target_speed - speed, 0.0, 0.25)
		throttle = self._speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, 0.75)

		if brake:
			steer *= 0.5
			throttle = 0.0

		return steer, throttle, self.should_brake, target_speed

	def get_transfuser_control(self, tick_data):
		pos = self.get_transfuser_position(tick_data)
		tick_data['gps'] = pos
		
		next_wp, next_cmd = self._route_planner.run_step(pos)
		tick_data['next_command'] = next_cmd.value

		theta = tick_data['compass'] + np.pi/2

		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		tick_data['target_point'] = tuple(local_command_point)

		if self.step < self.config.seq_len:
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			
			if not self.config.ignore_sides:
				rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
				
				rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			if not self.config.ignore_rear:
				rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

			self.input_buffer['lidar'].append(tick_data['lidar'])
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].append(tick_data['compass'])

			transfuser_control = carla.VehicleControl()
			transfuser_control.steer = 0.0
			transfuser_control.throttle = 0.0
			transfuser_control.brake = 0.0

			transfuser_waypoints = {
				'near_node': pos,
				'far_node': next_wp
				}

			return transfuser_control, transfuser_waypoints
		
		else:
			gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
			command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

			tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]), torch.FloatTensor([tick_data['target_point'][1]])]
			target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

			encoding = []
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb'].popleft()
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			
			if not self.config.ignore_sides:
				rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_left'].popleft()
				self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
				
				rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_right'].popleft()
				self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			if not self.config.ignore_rear:
				rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_rear'].popleft()
				self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

			self.input_buffer['lidar'].popleft()
			self.input_buffer['lidar'].append(tick_data['lidar'])
			self.input_buffer['gps'].popleft()
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].popleft()
			self.input_buffer['thetas'].append(tick_data['compass'])

			# transform the lidar point clouds to local coordinate frame
			ego_theta = self.input_buffer['thetas'][-1]
			ego_x, ego_y = self.input_buffer['gps'][-1]

			# only predict every second step because we only get a LiDAR every second frame.
			if(self.step  % 2 == 0 or self.step <= 4):
				for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
					curr_theta = self.input_buffer['thetas'][i]
					curr_x, curr_y = self.input_buffer['gps'][i]
					
					lidar_point_cloud[:,1] *= -1 # inverts x, y
					lidar_transformed = transform_2d_points(lidar_point_cloud, np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
					lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
					
					self.lidar_processed = list()
					self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))

				self.pred_wp = self.net(self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
								self.input_buffer['rgb_right']+self.input_buffer['rgb_rear'], \
								self.lidar_processed, target_point, gt_velocity)

			steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
			self.pid_metadata = metadata

			if brake < 0.05:
				brake = 0.0
			if throttle > brake:
				brake = 0.0

			transfuser_control = carla.VehicleControl()
			transfuser_control.steer = float(steer)
			transfuser_control.throttle = float(throttle)
			transfuser_control.brake = float(brake)

			if SAVE_DATA:
				cv2.imshow("Lidar", cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True))
				if self.step % 10 == 0 and self.save_path is not None:
					self.transfuser_save(tick_data)

			transfuser_waypoints = {
				'near_node': [pos[0] + metadata['wp_1'][0], pos[1] + metadata['wp_1'][1]],
				'far_node': [pos[0] + metadata['wp_2'][0], pos[1] + metadata['wp_2'][1]]
				}

			return transfuser_control, transfuser_waypoints

	def tick(self, input_data):
		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		
		# it can happen that the compass sends nan for a few frames
		if (math.isnan(compass) == True):
			compass = 0.0

		lidar = input_data['lidar'][1][:, :3]

		weather = weather_to_dict(input_data['weather'])

		result = {
				'rgb': rgb,
				'rgb_left': rgb_left,
				'rgb_right': rgb_right,
				'rgb_rear': rgb_rear,
				'lidar': lidar,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'weather': weather
				}

		return result

	def auto_pilot_run_step(self, tick_data):
		pos = self.get_auto_pilot_position(tick_data)

		near_node, near_command = self._waypoint_planner.run_step(pos)
		far_node, far_command = self._command_planner.run_step(pos)
		
		steer, throttle, brake, target_speed = self.get_auto_pilot_control(near_node, far_node, tick_data)

		speed = tick_data['speed']
		theta = tick_data['compass']
		weather = tick_data['weather']
		
		data_measurements = {
			'x': pos[0],
            'y': pos[1],
            'theta': theta,
            'speed': speed,
            'target_speed': target_speed,
            'x_command': far_node[0],
            'y_command': far_node[1],
            'command': near_command.value,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'weather': weather,
            'weather_id': self.weather_id,
            'near_node_x': near_node[0],
            'near_node_y': near_node[1],
            'far_node_x': far_node[0],
            'far_node_y': far_node[1],
            'should_slow': self.should_slow,
            'should_brake': self.should_brake,
            'angle': self.angle,
            'angle_unnorm': self.angle_unnorm,
            'angle_far_unnorm': self.angle_far_unnorm,
            }

		#print("steer : ", steer, "angle : ", self.angle, "theta : ", theta, "pos : ",pos, \
		#	"near_node : ", near_node, "near_command : ", near_command, "far_node : ", far_node, "far_command : ", far_command)
		 
		control = carla.VehicleControl()
		control.steer = steer + 1e-2 * np.random.randn()
		control.throttle = throttle
		control.brake = float(brake)
		
		if SAVE_DATA:
			if self.step % 10 == 0 and self.step != 0 and self.save_path is not None:
				self.auto_pilot_save(data_measurements, tick_data)
			
		return control, data_measurements

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		global dagger_index
		global dataset_count
		global total_steps

		if not self.initialized:
			self.init()

		self.step += 1
		total_steps += 1

		# generate random weather at each 25 steps
		if self.step % 25 == 0:
			index = random.choice(range(len(WEATHERS)))
			
			self.weather_id = WEATHERS_IDS[index]
			weather = WEATHERS[WEATHERS_IDS[index]]
			self._world.set_weather(weather)

		# add weather states to the input data
		input_data['weather'] = self._world.get_weather()

		# only one tick function for both autopilot and transfuser agent
		tick_data = self.tick(input_data)

		# control outputs for autopilot and transfuser motions
		auto_pilot_control, data_measurements = self.auto_pilot_run_step(tick_data)
		transfuser_control, meta_data = self.get_transfuser_control(tick_data)

		# DAgger metric
		# is_dagger = dagger_metric_action(auto_pilot_control, transfuser_control)
		is_dagger, distance_error = dagger_metric_waypoint(meta_autopilot=data_measurements, meta_transfuser=meta_data)
		
		image_bgr = cv2.cvtColor(tick_data['rgb'], cv2.COLOR_RGB2BGR)
		cv2.imshow("Front Image", image_bgr)
		cv2.waitKey(1)
		
		if self.step % 10 == 0:
			print("time:", round(timestamp, 2), "dagger status:", is_dagger, "total steps:", total_steps, "step:", self.step, "error:", round(distance_error, 3), \
				"---EXPERT: ", auto_pilot_control.throttle, round(auto_pilot_control.steer, 2), auto_pilot_control.brake, \
				"---AGENT: ", transfuser_control.throttle, round(transfuser_control.steer, 2), transfuser_control.brake)

		# count how many steps an expert agent takes brake action
		if auto_pilot_control.brake:
			self.brake_count += 1
		else:
			self.brake_count = 0

		# dynamic plot
		self.plot_info(auto_pilot_control, transfuser_control, total_steps)

		# append dataset
		if is_dagger:
			if self.step % 10 == 0 and self.brake_count < BRAKE_LIMIT and self.dagger_path is not None:
				print("[ DAgger Saved ! ]", "dataset number: ", dataset_count, "dagger count: ", dagger_index)

				dagger_index += 1
				self.dagger_save(data_measurements, tick_data, dagger_index)

			if dagger_index >= DATASET_LIMIT:
				dagger_index = 0
				dataset_count += 1

				self.setup()

			return auto_pilot_control

		# continue without appending the dataset
		else:
			return transfuser_control

	def get_forward_speed(self, transform=None, velocity=None):
		if not velocity:
			velocity = self._vehicle.get_velocity()
		
		if not transform:
			transform = self._vehicle.get_transform()
			
		vel_np = np.array([velocity.x, velocity.y, velocity.z])
		pitch = np.deg2rad(transform.rotation.pitch)
		yaw = np.deg2rad(transform.rotation.yaw)
		orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
		speed = np.dot(vel_np, orientation)
		
		return speed

	def _should_brake(self):
		actors = self._world.get_actors()

		self._traffic_lights = get_nearby_lights(self._vehicle, actors.filter('*traffic_light*'))
		
		vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
		light = self._is_light_red(actors.filter('*traffic_light*'))
		walker = self._is_walker_hazard(actors.filter('*walker*'))
		stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))
		
		self._is_vehicle_present = 1 if vehicle is not None else 0
		self._is_red_light_present = 1 if light is not None else 0
		self._is_pedestrian_present = 1 if walker is not None else 0
		self._is_stop_sign_present = 1 if stop_sign is not None else 0
		
		return any(x is not None for x in [vehicle, light, walker, stop_sign])

	def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
		affected = False
        
		# first we run a fast coarse test
		current_location = actor.get_location()
		stop_location = stop.get_transform().location
        
		if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
			return affected

		stop_t = stop.get_transform()
		transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
		list_locations = [current_location]
		waypoint = self._world.get_map().get_waypoint(current_location)
        
		for _ in range(multi_step):
			if waypoint:
				waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
				if not waypoint:
				    break
				list_locations.append(waypoint.transform.location)

		for actor_location in list_locations:
		    if get_point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
		        affected = True

		return affected

	def _is_stop_sign_hazard(self, stop_sign_list):
		if self._affected_by_stop:
			
			if not self._stop_completed:
				current_speed = self.get_forward_speed()
				
				if current_speed < self.SPEED_THRESHOLD:
					self._stop_completed = True
					return None	
				else:
					return self._target_stop_sign
		
			else:
			    # reset if the ego vehicle is outside the influence of the current stop sign            
			    if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
			        self._affected_by_stop = False
			        self._stop_completed = False
			        self._target_stop_sign = None
            
			    return None

		ve_tra = self._vehicle.get_transform()
		ve_dir = ve_tra.get_forward_vector()

		wp = self._world.get_map().get_waypoint(ve_tra.location)
		wp_dir = wp.transform.get_forward_vector()

		dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

		# ignore all when going in a wrong lane
		if dot_ve_wp > 0:
			for stop_sign in stop_sign_list:

			    if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
			        self._affected_by_stop = True
			        self._target_stop_sign = stop_sign
			        return self._target_stop_sign

		return None

	def _is_light_red(self, lights_list):
		if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green: 
		    affecting = self._vehicle.get_traffic_light()

		    for light in self._traffic_lights:
		        if light.id == affecting.id:
		            return affecting

		return None

	def _is_walker_hazard(self, walkers_list):
		z = self._vehicle.get_location().z
		p1 = to_numpy(self._vehicle.get_location())
		v1 = 10.0 * to_orientation(self._vehicle.get_transform().rotation.yaw)
		
		for walker in walkers_list:
			v2_hat = to_orientation(walker.get_transform().rotation.yaw)
			s2 = np.linalg.norm(to_numpy(walker.get_velocity()))
			
			if s2 < 0.05:
				v2_hat *= s2
				
			p2 = -3.0 * v2_hat + to_numpy(walker.get_location())
			v2 = 8.0 * v2_hat
			
			collides, collision_point = get_collision(p1, v1, p2, v2)
			
			if collides:
				return walker
				
		return None
		
	def _is_vehicle_hazard(self, vehicle_list):
		z = self._vehicle.get_location().z
		
		o1 = to_orientation(self._vehicle.get_transform().rotation.yaw)
		p1 = to_numpy(self._vehicle.get_location())
		
		s1 = max(10, 3.0 * np.linalg.norm(to_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
		v1_hat = o1
		v1 = s1 * v1_hat
		
		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._vehicle.id:
				continue
			
			o2 = to_orientation(target_vehicle.get_transform().rotation.yaw)
			p2 = to_numpy(target_vehicle.get_location())
			s2 = max(5.0, 2.0 * np.linalg.norm(to_numpy(target_vehicle.get_velocity())))
			
			v2_hat = o2
			v2 = s2 * v2_hat
			
			p2_p1 = p2 - p1
			distance = np.linalg.norm(p2_p1)
			p2_p1_hat = p2_p1 / (distance + 1e-4)
			
			angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
			angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
			angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
			angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
			
			if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
				continue
			elif angle_to_car > 30.0:
				continue
			elif distance > s1:
				continue
			
			return target_vehicle
			
		return None
        
	def transfuser_save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'transfuser' / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(self.save_path / 'transfuser' / 'lidar_0' / ('%04d.png' % frame))
		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(self.save_path / 'transfuser' / 'lidar_1' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'transfuser' / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def auto_pilot_save(self, data_measurements, tick_data):
		frame = self.step // 10
		
		measurements_file = self.save_path / 'autopilot' / 'measurements' / ('%04d.json' % frame)
		f = open(measurements_file, 'w')
		json.dump(data_measurements, f, indent=4)
		f.close()
		
		Image.fromarray(tick_data['rgb']).save(self.save_path / 'autopilot' / 'rgb' / ('%04d.png' % frame))
		for pos in ['left', 'right', 'rear']:
			name = 'rgb_' + pos
			Image.fromarray(tick_data[name]).save(self.save_path / 'autopilot' / name / ('%04d.png' % frame))
		
		np.save(self.save_path / 'autopilot' / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)

	def dagger_save(self, data_measurements, tick_data, dagger_index):
		Image.fromarray(tick_data['rgb']).save(self.dagger_path / self.dataset_name / 'rgb' / ('%04d.png' % dagger_index))

		for pos in ['left', 'right', 'rear']:
			name = 'rgb_' + pos
			Image.fromarray(tick_data[name]).save(self.dagger_path / self.dataset_name / name / ('%04d.png' % dagger_index))

		np.save(self.dagger_path / self.dataset_name / 'lidar' / ('%04d.npy' % dagger_index), tick_data['lidar'], allow_pickle=True)

		measurements_file = self.dagger_path / self.dataset_name / 'measurements' / ('%04d.json' % dagger_index)
		f = open(measurements_file, 'w')
		json.dump(data_measurements, f, indent=4)
		f.close()

	def destroy(self):
		del self.net
	
	def plot_info(self, auto_pilot_control, transfuser_control, total_steps):
		self.log_writer.add_scalars("throttle", {
			"expert": auto_pilot_control.throttle,
			"transfuser": transfuser_control.throttle,
		}, total_steps)
		
		self.log_writer.add_scalars("steer", {
			"expert": auto_pilot_control.steer,
			"transfuser": transfuser_control.steer,
		}, total_steps)

		self.log_writer.add_scalars("brake", {
			"expert": auto_pilot_control.brake,
			"transfuser": transfuser_control.brake,
		}, total_steps)
