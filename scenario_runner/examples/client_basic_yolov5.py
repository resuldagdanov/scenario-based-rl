#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import re
import time

from torch._C import dtype
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import coco_classes
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_c
    from pygame.locals import K_r
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
import torch
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 60

BB_COLOR = (248, 64, 24)
YOLO_COLOR = (64, 248, 24)
# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================

max_value = 255
max_value_H = 360//2
low_H_1 = 0
high_H_1 = 43 
low_S_1 = 152
high_S_1 = 255
low_V_1 = 151
high_V_1 = 255

low_H_2 = 43
high_H_2 = 139 
low_S_2 = 113
high_S_2 = 255
low_V_2 = 151
high_V_2 = 255


def on_low_H_1_thresh_trackbar(val):
    global low_H_1
    global high_H_1
    low_H_1 = val
    low_H_1 = min(high_H_1-1, low_H_1)
    cv2.setTrackbarPos("box1_lowH", "box1", low_H_1)

def on_high_H_1_thresh_trackbar(val):
    global low_H_1
    global high_H_1
    high_H_1 = val
    high_H_1 = max(high_H_1, low_H_1+1)
    cv2.setTrackbarPos("box1_highH", "box1", high_H_1)

def on_low_S_1_thresh_trackbar(val):
    global low_S_1
    global high_S_1
    low_S_1 = val
    low_S_1 = min(high_S_1-1, low_S_1)
    cv2.setTrackbarPos("box1_lowS", "box1", low_S_1)

def on_high_S_1_thresh_trackbar(val):
    global low_S_1
    global high_S_1
    high_S_1 = val
    high_S_1 = max(high_S_1, low_S_1+1)
    cv2.setTrackbarPos("box1_highS", "box1", high_S_1)

def on_low_V_1_thresh_trackbar(val):
    global low_V_1
    global high_V_1
    low_V_1 = val
    low_V_1 = min(high_V_1-1, low_V_1)
    cv2.setTrackbarPos("box1_lowV", "box1", low_V_1)

def on_high_V_1_thresh_trackbar(val):
    global low_V_1
    global high_V_1
    high_V_1 = val
    high_V_1 = max(high_V_1, low_V_1+1)
    cv2.setTrackbarPos("box1_highV", "box1", high_V_1)

def on_low_H_2_thresh_trackbar(val):
    global low_H_2
    global high_H_2
    low_H_2 = val
    low_H_2 = min(high_H_2-1, low_H_2)
    cv2.setTrackbarPos("box2_lowH", "box2", low_H_2)

def on_high_H_2_thresh_trackbar(val):
    global low_H_2
    global high_H_2
    high_H_2 = val
    high_H_2 = max(high_H_2, low_H_2+1)
    cv2.setTrackbarPos("box2_highH", "box2", high_H_2)

def on_low_S_2_thresh_trackbar(val):
    global low_S_2
    global high_S_2
    low_S_2 = val
    low_S_2 = min(high_S_2-1, low_S_2)
    cv2.setTrackbarPos("box2_lowS", "box2", low_S_2)

def on_high_S_2_thresh_trackbar(val):
    global low_S_2
    global high_S_2
    high_S_2 = val
    high_S_2 = max(high_S_2, low_S_2+1)
    cv2.setTrackbarPos("box2_highS", "box2", high_S_2)

def on_low_V_2_thresh_trackbar(val):
    global low_V_2
    global high_V_2
    low_V_2 = val
    low_V_2 = min(high_V_2-1, low_V_2)
    cv2.setTrackbarPos("box2_lowV", "box2", low_V_2)

def on_high_V_2_thresh_trackbar(val):
    global low_V_2
    global high_V_2
    high_V_2 = val
    high_V_2 = max(high_V_2, low_V_2+1)
    cv2.setTrackbarPos("box2_highV", "box2", high_V_2)

cv2.namedWindow("box1")
cv2.namedWindow("box2")

cv2.createTrackbar("box1_lowH", "box1" , low_H_1, max_value_H, on_low_H_1_thresh_trackbar)
cv2.createTrackbar("box1_highH", "box1" , high_H_1, max_value_H, on_high_H_1_thresh_trackbar)
cv2.createTrackbar("box1_lowS", "box1" , low_H_1, max_value, on_low_S_1_thresh_trackbar)
cv2.createTrackbar("box1_highS", "box1" , high_H_1, max_value, on_high_S_1_thresh_trackbar)
cv2.createTrackbar("box1_lowV", "box1" , low_H_1, max_value, on_low_V_1_thresh_trackbar)
cv2.createTrackbar("box1_highV", "box1" , high_H_1, max_value, on_high_V_1_thresh_trackbar)

cv2.createTrackbar("box2_lowH", "box2" , low_H_2, max_value_H, on_low_H_2_thresh_trackbar)
cv2.createTrackbar("box2_highH", "box2" , high_H_2, max_value_H, on_high_H_2_thresh_trackbar)
cv2.createTrackbar("box2_lowS", "box2" , low_H_2, max_value, on_low_S_2_thresh_trackbar)
cv2.createTrackbar("box2_highS", "box2" , high_H_2, max_value, on_high_S_2_thresh_trackbar)
cv2.createTrackbar("box2_lowV", "box2" , low_H_2, max_value, on_low_V_2_thresh_trackbar)
cv2.createTrackbar("box2_highV", "box2" , high_H_2, max_value, on_high_V_2_thresh_trackbar)


def get_light_state(image, x1, y1, x2, y2):
    global low_H_1, low_H_2, low_S_1, low_S_2, low_V_1, low_V_2
    global high_H_1, high_H_2, high_S_1, high_S_2, high_V_1, high_V_2 

    temp = int((y2 - y1 )/3)
    light_box = image[y1:y2, x1:x2, :]
    box1 = image[y1:y1+temp, x1:x2, :]
    box2 = image[y1+temp:y1+temp*2, x1:x2, :]
    box3 = image[y1+temp*2:y2, x1:x2, :]
    box1_hsv = cv2.cvtColor(box1, cv2.COLOR_RGB2HSV)
    box2_hsv = cv2.cvtColor(box2, cv2.COLOR_RGB2HSV)
    box3_hsv = cv2.cvtColor(box3, cv2.COLOR_RGB2HSV)
    light_hsv = cv2.cvtColor(light_box, cv2.COLOR_RGB2HSV)

    b1_threshold = cv2.inRange(light_hsv, (low_H_1, low_S_1, low_V_1), (high_H_1, high_S_1, high_V_1))
    b2_threshold = cv2.inRange(light_hsv, (low_H_2, low_S_2, low_V_2), (high_H_2, high_S_2, high_V_2))

    cv2.imshow("box1", np.array(light_box[:, :, ::-1], dtype=np.uint8))
    cv2.imshow("b1_thresh", b1_threshold)
    cv2.imshow("box2", np.array(light_box[:, :, ::-1], dtype=np.uint8))
    cv2.imshow("b2_thresh", b2_threshold)

    count = np.zeros(3)
    count[0] = np.count_nonzero(b1_threshold)
    count[1] = np.count_nonzero(b2_threshold)

    max = np.argmax(count)
    if max == 0:
        return "Red"
    elif max == 1:
        return "Green"

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))


    @staticmethod
    def draw_detections(display, detections, image):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for dets in detections[0]:
            x1 = int(dets[0]) + image.shape[1]//2
            y1 = int(dets[1])
            x2 = int(dets[2]) + image.shape[1]//2
            y2 = int(dets[3])
            conf = dets[4]
            if conf < 0.6:
                continue
            cls = coco_classes.class_id[int(dets[5] + 1)]
            if dets[5] == 9.0:
                light_state = get_light_state(image, x1, y1, x2, y2)
            text_surface = None
            if dets[5] == 9.0:
                text_surface = pygame.font.SysFont('Comic Sans MS', 16).render("%s %.3f"%(light_state, conf), True, YOLO_COLOR)
            elif dets[5] == 11.0:
                if dets[4] > 0.8:
                    text_surface = pygame.font.SysFont('Comic Sans MS', 16).render("%s %.3f"%("STOP", conf), True, YOLO_COLOR)
            else:
                text_surface = pygame.font.SysFont('Comic Sans MS', 16).render("%s %.3f"%(cls, conf), True, YOLO_COLOR)
            if text_surface is not None:
                bb_surface.blit(text_surface, (x1, y1 - 10))
            else:
                continue

            #print(dets)

            # draw lines
            # base
            pygame.draw.line(bb_surface, YOLO_COLOR, (x1, y1), (x1, y2))
            pygame.draw.line(bb_surface, YOLO_COLOR, (x1, y2), (x2, y2))
            pygame.draw.line(bb_surface, YOLO_COLOR, (x2, y2), (x2, y1))
            pygame.draw.line(bb_surface, YOLO_COLOR, (x2, y1), (x1, y1))

                
        
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords


    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True


        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        print(preset[1])
        self.world.set_weather(preset[0])

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        elif keys[K_c]:
            self.next_weather()
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self):
        """
        Main program loop.
        """
        map_name = "Town05"
        try:
            pygame.init()
            self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            map = self.world.get_map()

            if map.name != map_name:
                self.world = self.client.load_world(map_name)
                self.world = self.client.reload_world()
                time.sleep(5)

            self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

            self.yolo.cuda()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.bmv.*')


            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)

                array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (self.image.height, self.image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]

                
                results = self.yolo(array[:, array.shape[1]//2:, :])

                bounding_boxes = []
                if results.xyxy[0].size() != (0, 6):
                    ClientSideBoundingBoxes.draw_detections(self.display, results.xyxy, array)



                cv2.waitKey(1)
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
