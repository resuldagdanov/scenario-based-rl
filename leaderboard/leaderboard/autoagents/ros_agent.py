#!/usr/bin/env python3
#
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

"""
This module provides a ROS autonomous agent interface to control the ego vehicle via a ROS stack
"""

import math
import os
import subprocess
import signal
import threading
import time

import numpy

import carla
from scipy.spatial import distance
import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2, NavSatFix, NavSatStatus, CameraInfo, Imu
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseWithCovarianceStamped
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header, String
# pylint: disable=line-too-long
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleInfo, CarlaEgoVehicleInfoWheel, CarlaEgoVehicleControl, CarlaWorldInfo
from geometry_msgs.msg import Vector3Stamped
# pylint: enable=line-too-long
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

import gps_utils
import spline_planner

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
def get_entry_point():
    return 'RosAgent'



class RosAgent(AutonomousAgent):

    """
    Base class for ROS-based stacks.

    Derive from it and implement the sensors() method.

    Please define TEAM_CODE_ROOT in your environment.
    The stack is started by executing $TEAM_CODE_ROOT/start.sh

    The sensor data is published on similar topics as with the carla-ros-bridge. You can find details about
    the utilized datatypes there.

    This agent expects a roscore to be running.
    """
    init = True
    speed = None
    current_control = None
    stack_process = None
    timestamp = None
    current_map_name = None
    step_mode_possible = None
    vehicle_info_publisher = None
    global_plan_published = None
    imu_init = True
    def setup(self, path_to_conf_file):
        """
        setup agent
        """
        self.track = Track.SENSORS
        self.stack_thread = None

        # get start_script from environment
        '''
        team_code_path = os.environ['TEAM_CODE_ROOT']
        if not team_code_path or not os.path.exists(team_code_path):
            raise IOError("Path '{}' defined by TEAM_CODE_ROOT invalid".format(team_code_path))
        start_script = "{}/start.sh".format(team_code_path)
        if not os.path.exists(start_script):
            raise IOError("File '{}' defined by TEAM_CODE_ROOT invalid".format(start_script))
        '''
        #set use_sim_time via commandline before init-node
        # process = subprocess.Popen(
        #     "rosparam set use_sim_time true", shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        # process.wait()
        # if process.returncode:
        #     raise RuntimeError("Could not set use_sim_time")

        # initialize ros node
        rospy.init_node('ros_agent', anonymous=True)

        # publish first clock value '0'
        self.clock_publisher = rospy.Publisher('clock', Clock, queue_size=10, latch=True)
        self.clock_publisher.publish(Clock(rospy.Time.from_sec(0)))

        # execute script that starts the ad stack (remains running)
        rospy.loginfo("Executing stack...")
    #    self.stack_process = subprocess.Popen(start_script, shell=True, preexec_fn=os.setpgrp)

        self.vehicle_control_event = threading.Event()
        self.timestamp = None
        self.speed = 0
        self.global_plan_published = False

        self.vehicle_info_publisher = None
        self.vehicle_status_publisher = None
        self.odometry_publisher = None
        self.world_info_publisher = None
        self.map_file_publisher = None
        self.current_map_name = None
        self.tf_broadcaster = None
        self.lidar_data_cumulative = None
        self.step_mode_possible = False
        self.imu_publisher = None
        self.sparse_coords = None
        self.dense_msg = Path()
        self.initial_yaw_angle = None
        self.num_lidar = 0

        self.vehicle_control_subscriber = rospy.Subscriber(
            '/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, self.on_vehicle_control)

        self.current_control = carla.VehicleControl()

        self.waypoint_publisher = rospy.Publisher(
            '/eatron/global_planner/path', Path, queue_size=1, latch=True)

        self.waypoint_publisher_sampled = rospy.Publisher(
            '/eatron/global_planner/sampled_path', Path, queue_size=1, latch=True)

        self.publisher_map = {}
        self.id_to_sensor_type_map = {}
        self.id_to_camera_info_map = {}
        self.cv_bridge = CvBridge()

        # setup ros publishers for sensors
        # pylint: disable=line-too-long
        for sensor in self.sensors():
            self.id_to_sensor_type_map[sensor['id']] = sensor['type']
            if sensor['type'] == 'sensor.camera.rgb':
                self.publisher_map[sensor['id']] = rospy.Publisher(
                    '/carla/ego_vehicle/camera/rgb/' + sensor['id'] + "/image_color", Image, queue_size=1, latch=True)
                self.id_to_camera_info_map[sensor['id']] = self.build_camera_info(sensor)
                self.publisher_map[sensor['id'] + '_info'] = rospy.Publisher(
                    '/carla/ego_vehicle/camera/rgb/' + sensor['id'] + "/camera_info", CameraInfo, queue_size=1, latch=True)
            elif sensor['type'] == 'sensor.lidar.ray_cast':
                self.publisher_map[sensor['id']] = rospy.Publisher(
                    '/carla/ego_vehicle/lidar/' + sensor['id'] + "/point_cloud", PointCloud2, queue_size=1, latch=True)
            elif sensor['type'] == 'sensor.other.gnss':
                self.publisher_map[sensor['id']] = rospy.Publisher(
                    '/carla/ego_vehicle/gnss/' + sensor['id'] + "/fix", Odometry, queue_size=1, latch=True)
            elif sensor['type'] == 'sensor.other.imu':
                self.publisher_map[sensor['id']] = rospy.Publisher('/carla/ego_vehicle/imu/data', Imu, queue_size=1, latch=True)
            elif sensor['type'] == 'sensor.speedometer':
                self.publisher_map[sensor['id']] = rospy.Publisher('/carla/ego_vehicle/speed', Odometry, queue_size=1, latch=True)
            elif sensor['type'] == 'sensor.other.radar':
                if sensor['id'] == 'RADAR_REAR':
                    self.publisher_map[sensor['id']] = rospy.Publisher('/carla/ego_vehicle/radar_rear', PointCloud2, queue_size=1, latch=True)
                else:
                    self.publisher_map[sensor['id']] = rospy.Publisher('/carla/ego_vehicle/radar_front', PointCloud2, queue_size=1, latch=True)

            else:
                raise TypeError("Invalid sensor type: {}".format(sensor['type']))
        # pylint: enable=line-too-long

    def destroy(self):
        """
        Cleanup of all ROS publishers
        """
        if self.stack_process and self.stack_process.poll() is None:
            rospy.loginfo("Sending SIGTERM to stack...")
            os.killpg(os.getpgid(self.stack_process.pid), signal.SIGTERM)
            rospy.loginfo("Waiting for termination of stack...")
            self.stack_process.wait()
            time.sleep(5)
            rospy.loginfo("Terminated stack.")

        rospy.loginfo("Stack is no longer running")
        #self.world_info_publisher.unregister()
        #self.map_file_publisher.unregister()
        #self.vehicle_status_publisher.unregister()
        #self.vehicle_info_publisher.unregister()
        self.waypoint_publisher.unregister()
        self.stack_process = None
        rospy.loginfo("Cleanup finished")

    def on_vehicle_control(self, data):
        """
        callback if a new vehicle control command is received
        """
        cmd = carla.VehicleControl()
        cmd.throttle = data.throttle
        cmd.steer = data.steer
        cmd.brake = data.brake
        cmd.hand_brake = data.hand_brake
        cmd.reverse = data.reverse
        cmd.gear = data.gear
        cmd.manual_gear_shift = data.manual_gear_shift
        self.current_control = cmd
        if not self.vehicle_control_event.is_set():
            self.vehicle_control_event.set()
        # After the first vehicle control is sent out, it is possible to use the stepping mode
        self.step_mode_possible = True

    def build_camera_info(self, attributes):  # pylint: disable=no-self-use
        """
        Private function to compute camera info

        camera info doesn't change over time
        """
        camera_info = CameraInfo()
        # store info without header
        camera_info.header = None
        camera_info.width = int(attributes['width'])
        camera_info.height = int(attributes['height'])
        camera_info.distortion_model = 'plumb_bob'
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        fx = camera_info.width / (
            2.0 * math.tan(float(attributes['fov']) * math.pi / 360.0))
        fy = fx
        camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info.D = [0, 0, 0, 0, 0]
        camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]
        return camera_info



    def publish_plan(self):
        """
        publish the global plan
        """
    
        latitude_0 = self._global_plan[0][0]["lat"]
        longitude_0 = self._global_plan[0][0]["lon"]
        altitude_0 = self._global_plan[0][0]["z"]
        msg = Path()
        msg.header.frame_id = "/map_positive"
        msg.header.stamp = rospy.Time.now()
        curr_sparse_coords = []
        dense_coords_new = []
        for gp in self._global_plan:
            #print(gp[0])
            pose = PoseStamped()


            x,y,z = gps_utils.geodetic_to_ecef(gp[0]["lat"],gp[0]["lon"],gp[0]["z"])
            x,y,z = gps_utils.ecef_to_enu(x,y,z, latitude_0,longitude_0,altitude_0)


            if self.initial_yaw_angle and self.initial_yaw_angle != 0:
                c = math.cos(self.initial_yaw_angle)
                s = math.sin(self.initial_yaw_angle)
                R = numpy.array([[c, -s], [s, c,]])
                temp_x = x
                x = c * x + s * y
                y = c * y - s * temp_x
            pose.pose.position.x = x + 10000
            pose.pose.position.y = y + 10000
            

            curr_sparse_coords.append([x, y])

            pose.pose.position.z = z
        

            msg.poses.append(pose)

        if self.sparse_coords != curr_sparse_coords or self.sparse_coords == None:
            #dense_coords = spline_planner.calc_spline_course(curr_sparse_coords, ds = 1.5)
            dense_coords = gps_utils.interpolation(curr_sparse_coords, ds = 1.5)
            dense_coords = numpy.array(dense_coords)
            dense_msg = Path()
            dense_msg.header.frame_id = "/map_positive"
            dense_msg.header.stamp = rospy.Time.now()
            for k in range(len(dense_coords)-1):
                dense_stamped = PoseStamped()
                dense_stamped.pose.position.x = dense_coords[k,0] + 10000
                dense_stamped.pose.position.y = dense_coords[k,1] + 10000

                dense_stamped.pose.position.z = 0
                dense_msg.poses.append(dense_stamped)
            self.dense_msg = dense_msg

        rospy.loginfo("Publishing Plan...")
        self.sparse_coords = curr_sparse_coords
        self.waypoint_publisher.publish(msg)
        self.waypoint_publisher_sampled.publish(self.dense_msg)

    def interpolation(self,coords, threshold):
        _coords = numpy.array(coords)

        x_coords =_coords[:, 0]
        y_coords = _coords[:, 1]

        dense_coords = []

        for i in range(len(coords) - 1):
            # checking distance between each points
            dist = distance.euclidean(_coords[i], _coords[i + 1])
            if (dist > threshold):
                # adding number of required data points for interpolation
                ratio = dist / float(threshold) 
                x_step = (x_coords[i + 1] - x_coords[i]) / ratio
                y_step = (y_coords[i + 1] - y_coords[i]) / ratio

                for j in range(int(round(ratio))):
                    dense_coords.append([x_coords[i] + (j * x_step), y_coords[i] + (j * y_step)])

        _dense_coords = numpy.array(dense_coords)
        return _dense_coords

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 800, 'height': 600, 'fov': 60, 'id': 'Center'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -90 , 'id': 'LIDAR'},
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
            {'type': 'sensor.other.imu', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -45.0, 'id': 'IMU'},
            # {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
            {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
            {'type': 'sensor.other.radar', 'x': 1.7, 'y': 0, 'z': 1, 'roll': 0.0, 'pitch': 0.0,
             'yaw': 0, 'fov': 30, 'id': 'RADAR'},
            {'type': 'sensor.other.radar', 'x': -1.7, 'y': 0, 'z': 1, 'roll': 0.0, 'pitch': 0.0,
             'yaw': -180, 'fov': 30, 'id': 'RADAR_REAR'},

           
        ]

        return sensors
    

    def publish_tf(self):

        inverse_frame = -1


        br = tf.TransformBroadcaster()
        for sensor in self.sensors():
            if sensor['type'] == 'sensor.other.gnss' or sensor['type'] == 'sensor.speedometer':
                continue
                      
            br.sendTransform((sensor['x'], inverse_frame * sensor['y'], sensor['z']),
                tf.transformations.quaternion_from_euler(0, 0, inverse_frame * sensor['yaw']*math.pi/180),
                rospy.Time.now(),
                self.frames[sensor['id']],
                "base_link")

        br.sendTransform((0, 0, 0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "map_positive",
            "odom")
        


    def get_header(self):
        """
        Returns ROS message header
        """
        header = Header()
        #header.stamp = rospy.Time.from_sec(self.timestamp)
        header.stamp = rospy.Time.now()
        return header

    def publish_lidar(self, sensor_id, data):
        """
        Function to publish lidar data
        """
        header = self.get_header()
        header.frame_id = self.frames[sensor_id]
        lidar_data = numpy.frombuffer(
            data, dtype=numpy.float32)
  
        lidar_data = numpy.reshape(
            lidar_data, (int((lidar_data.shape[0]) / 4), 4))
        # we take the oposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        # we need a copy here, because the data are read only in carla numpy
        # array
        #lidar_data = -1.0 * lidar_data
        # we also need to permute x and y
        lidar_data = lidar_data[..., [0, 1, 2]]
        lidar_data[:,1] = -lidar_data[:,1]
        # R = numpy.identity(3)
        # if self.initial_yaw_angle:
        #     c = math.cos(self.initial_yaw_angle)
        #     s = math.sin(self.initial_yaw_angle)
        #     R = numpy.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if(self.num_lidar == 0):
            self.lidar_data_cumulative = lidar_data[:,0:3]
        else:
            self.lidar_data_cumulative = numpy.concatenate((self.lidar_data_cumulative, lidar_data[:,0:3]))
            msg = create_cloud_xyz32(header,self.lidar_data_cumulative)
            self.publisher_map[sensor_id].publish( msg)
            self.num_lidar = -1

        self.num_lidar +=1
 
        



    

    def publish_gnss(self, sensor_id, data):
        """
        Function to publish gnss data
        """
        #print(data)
        # msg = NavSatFix()
        # msg.header = self.get_header()
        # msg.header.frame_id = 'map'
        # msg.latitude = data[0]
        # msg.longitude = data[1]
        # msg.altitude = data[2]
        # msg.status.status = NavSatStatus.STATUS_SBAS_FIX
        # # pylint: disable=line-too-long
        # msg.status.service = NavSatStatus.SERVICE_GPS | NavSatStatus.SERVICE_GLONASS | NavSatStatus.SERVICE_COMPASS | NavSatStatus.SERVICE_GALILEO
        # # pylint: enable=line-too-long
        # self.publisher_map[sensor_id].publish(msg)

        latitude = data[0]
        longitude = data[1]
        altitude = data[2]
        odometry = Odometry()
        odometry.header = self.get_header()
        odometry.header.frame_id = 'odom'
        odometry.child_frame_id = 'base_link'


        
        if self.init:
            self.latitude_0 = self._global_plan[0][0]["lat"]
            self.longitude_0 = self._global_plan[0][0]["lon"]
            self.altitude_0 = self._global_plan[0][0]["z"]

            lat_dev = 0.000005
            lon_dev = 0.000005
            alt_dev = 0.000005

            self.x_dev,self.y_dev,self.z_dev = gps_utils.geodetic_to_ecef(lat_dev,lon_dev,alt_dev)
            self.x_dev,self.y_dev,self.z_dev = gps_utils.ecef_to_enu(self.x_dev,self.y_dev,self.z_dev,0,0,0)

            self.init = False

        x,y,z = gps_utils.geodetic_to_ecef(latitude,longitude,altitude)
        x,y,z = gps_utils.ecef_to_enu(x,y,z, self.latitude_0,self.longitude_0,self.altitude_0)
        
        # X-Y reversed to match odom frame

        if self.initial_yaw_angle and self.initial_yaw_angle != 0:
            c = math.cos(self.initial_yaw_angle)
            s = math.sin(self.initial_yaw_angle)
            R = numpy.array([[c, -s], [s, c,]])
            temp_x = x
            x = c * x + s * y
            y = c * y - s * temp_x
        odometry.pose.pose.position.x = x + 10000
        odometry.pose.pose.position.y = y + 10000
        odometry.pose.pose.position.z = z

        odometry.pose.covariance[0] = (self.y_dev**2) * 10
        odometry.pose.covariance[7] = (self.x_dev**2) * 10
        odometry.pose.covariance[14] = (self.z_dev**2) * 1e11

        self.publisher_map[sensor_id].publish(odometry)


    def publish_camera(self, sensor_id, data):
        """
        Function to publish camera data
        """
        msg = self.cv_bridge.cv2_to_imgmsg(data, encoding='bgra8')
        # the camera data is in respect to the camera's own frame
        msg.header = self.get_header()
        msg.header.frame_id = self.frames[sensor_id]

        cam_info = self.id_to_camera_info_map[sensor_id]
        cam_info.header = msg.header
        self.publisher_map[sensor_id + '_info'].publish(cam_info)
        self.publisher_map[sensor_id].publish(msg)
  


    def publish_imu(self, sensor_id, data):
        imu_message = Imu()
        imu_message.header = self.get_header()
        imu_message.header.frame_id = 'base_link'
        imu_message.linear_acceleration.x  = data[0]
        imu_message.linear_acceleration.y  = -data[1]
        imu_message.linear_acceleration.z  = 0
        imu_message.angular_velocity.x = data[3]
        imu_message.angular_velocity.y = data[4]
        imu_message.angular_velocity.z = -data[5]
        #add covariance to IMU
        imu_message.angular_velocity_covariance[0] = 1e-6
        imu_message.angular_velocity_covariance[4] = 1e-6
        imu_message.angular_velocity_covariance[8] = 1e-6

        imu_message.linear_acceleration_covariance[0] = 1e-6
        imu_message.linear_acceleration_covariance[4] = 1e-6
        imu_message.linear_acceleration_covariance[8] = 0.000225

        if self.imu_init:
        #     init_publisher = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=1, latch=True)
        #     initial_pose = PoseWithCovarianceStamped()
        #     initial_pose.header.frame_id = 'base_link'
            self.initial_yaw_angle = math.atan2(self.dense_msg.poses[3].pose.position.y - self.dense_msg.poses[1].pose.position.y,
                self.dense_msg.poses[3].pose.position.x - self.dense_msg.poses[1].pose.position.x)

            # quat = tf.transformations.quaternion_from_euler(0, 0, self.initial_yaw_angle)
            
            # initial_pose.pose.pose.orientation.x = quat[0]
            # initial_pose.pose.pose.orientation.y = quat[1]
            # initial_pose.pose.pose.orientation.z = quat[2]
            # initial_pose.pose.pose.orientation.w = quat[3]
            
            # init_publisher.publish(initial_pose)
            self.imu_init = False

        self.publisher_map[sensor_id].publish(imu_message)



    def use_stepping_mode(self):  # pylint: disable=no-self-use
        """
        Overload this function to use stepping mode!
        """
        return False


    def publish_speedometer(self, sensor_id, data):
        # speed_data = Vector3Stamped()
        # speed_data.header.stamp = rospy.Time.from_sec(self.timestamp)
        # speed_data.vector.x = data['speed']
        # self.publisher_map[sensor_id].publish(speed_data)

        odometry = Odometry()
        odometry.header = self.get_header()
        odometry.header.frame_id = 'base_link'
        odometry.twist.twist.linear.x = data['speed']/3.6
        odometry.twist.twist.linear.y = 0
        odometry.twist.twist.linear.z = 0
        self.publisher_map[sensor_id].publish(odometry)

    def publish_radar(self,sensor_id, data):
        msg = Float32MultiArray()
        header = self.get_header()
        header.frame_id = self.frames[sensor_id]
        radar_data = numpy.array(data, dtype=numpy.float32)

        #print(radar_data)
        # [depth, altitute, azimuth, velocity]
        depth = radar_data[:,0]
        altitude =  radar_data[:,1]
        azimuth =  radar_data[:,2]

        points_x = (depth * numpy.cos(azimuth) * numpy.cos(-altitude)).reshape(-1,1)
        points_y = (depth * numpy.sin(-azimuth) * numpy.cos(altitude)).reshape(-1,1)
        points_z = (depth * numpy.sin(altitude)).reshape(-1,1)
        
        # points_x = (depth * numpy.cos(azimuth) * numpy.cos(-altitude)).reshape(-1,1)
        # points_y = (depth * numpy.sin(altitude)).reshape(-1,1)
        # points_z = -1 * (depth * numpy.sin(-azimuth) * numpy.cos(altitude)).reshape(-1,1)


        radar_data = numpy.append(points_x,points_y,axis = 1)

        radar_data = numpy.append(radar_data, points_z,axis = 1)


        msg = create_cloud_xyz32(header,radar_data)
        # we take the oposite of y axis
        # (as lidar point are express in left handed coordinate system, and ros need right handed)
        # we need a copy here, because the data are read only in carla numpy
        # array
        #lidar_data = -1.0 * lidar_data
        # we also need to permute x and y

        
        #msg.data = data
        self.publisher_map[sensor_id].publish(msg)
        


    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self.vehicle_control_event.clear()
        self.timestamp = timestamp
        self.clock_publisher.publish(Clock(rospy.Time.from_sec(timestamp)))

        # check if stack is still running
        if self.stack_process and self.stack_process.poll() is not None:
            raise RuntimeError("Stack exited with: {} {}".format(
                self.stack_process.returncode, self.stack_process.communicate()[0]))

        # publish global plan to ROS once
        if self._global_plan_world_coord:
            self.publish_plan()

        new_data_available = False

        self.publish_tf()

        # publish data of all sensors
        for key, val in input_data.items():
            new_data_available = True
            sensor_type = self.id_to_sensor_type_map[key]
            if sensor_type == 'sensor.camera.rgb':
                self.publish_camera(key, val[1])
            elif sensor_type == 'sensor.lidar.ray_cast':
                self.publish_lidar(key, val[1])
            elif sensor_type == 'sensor.other.gnss':
                self.publish_gnss(key, val[1])
            elif sensor_type == 'sensor.other.imu':
                self.publish_imu(key,val[1])
            elif sensor_type == 'sensor.speedometer':
                self.publish_speedometer(key,val[1])
            elif sensor_type == 'sensor.other.radar':
                self.publish_radar(key,val[1])
            else:
                raise TypeError("Invalid sensor type: {}".format(sensor_type))

        if self.use_stepping_mode():
            if self.step_mode_possible and new_data_available:
                self.vehicle_control_event.wait()
        # if the stepping mode is not used or active, there is no need to wait here

        return self.current_control

