import cv2
import torch
import copy
import numpy as np

from eatron.utils import RangeTrackbar

class TrafficLightSignDetector:
    def __init__(self, conf_limit = 0.6, model_name = 'yolov5m', debug = False, det_visualize = True):
        #Get params
        self.conf_limit = conf_limit
        self.debug = debug
        self.det_visualize = det_visualize

        #Create YOLOv5
        self.yolo_det = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        if torch.cuda.is_available():
            self.yolo_det.cuda()

        #Traffic light HSV thresholds
        self.red_range_1 = [(0, 152, 151), (43, 255, 255)]
        self.red_range_2 = [(150, 152, 151), (180, 255, 255)]
        self.green_range = [(43, 113, 151), (150, 255, 255)]
        self.initial_traffic_light_ranges = [self.red_range_1, 
                                             self.red_range_2,
                                             self.green_range]

        self.stop_sign_range = [(0, 130, 0), (180, 255, 255)]
        self.other_sign_range = [(14, 0, 0), (168, 49, 239)]
        self.initial_traffic_sign_ranges = [self.stop_sign_range, 
                                             self.other_sign_range]

        #Image buffer
        self.image = []
        self.bboxes = []
        self.class_names = {
            0: "red",
            1: "green",
            2: "stop",
            3: "other"
        }
        if self.debug is True:

            self.image_det = []
            #Create trackbars for traffic light
            self.light_names = ["red1", "red2", "green"]
            self.light_bars = []
            for i, name in enumerate(self.light_names):
                cv2.namedWindow(name)
                self.light_bars.append(RangeTrackbar(name))
                self.light_bars[i].set_range(self.initial_traffic_light_ranges[i])

            #Create trackbars for signs
            self.sign_names = ["stop", "other"]
            self.sign_bars = []
            for i, name in enumerate(self.sign_names):
                cv2.namedWindow(name)
                self.sign_bars.append(RangeTrackbar(name))
                self.sign_bars[i].set_range(self.initial_traffic_sign_ranges[i])
        if self.det_visualize is True:
            self.detections_window = "detections"
            cv2.namedWindow(self.detections_window)

    #image = numpy array RGB 
    def inference(self, image, ratio=0.5):
        #Inference
        self.image = image
        self.bboxes = []
        results = self.yolo_det(image)

        if results.xyxy[0].size() != (0, 6):
            self.detections = results.xyxy[0].cpu().numpy()
            self.process_detections()

        #If debug is True show detections
        if self.det_visualize is True:
            self.draw_bboxes()
            cv2.imshow(self.detections_window, self.image_det[:, :, ::-1])
            cv2.waitKey(1)

        return self.bboxes

    #Process detections and create bboxes for traffic light and signs
    def process_detections(self):
        for det in self.detections:
            #confidence is high enough
            if det[4] > self.conf_limit:
                new_det = self._create_empty_box()
                #Check if the det is traffic light
                new_det['x1'] = int(det[0])
                new_det['x2'] = int(det[2])
                new_det['y1'] = int(det[1])
                new_det['y2'] = int(det[3])
                new_det['conf'] = det[4]
                
                if det[5] == 9.0:
                    new_det['c'] = self._get_light_state(new_det)

                #Check if traffic sign
                elif det[5] == 11.0:
                    new_det['c'] = self._get_sign(new_det)

                if new_det['c'] != -1:
                    self.bboxes.append(new_det)
        return self.bboxes
    
    #Create empty bounding box
    def _create_empty_box(self):
        new_det = {
            'x1': int(0),
            'x2': int(0),
            'y1': int(0),
            'y2': int(0),
            'c' : -1,
            'conf': 0.0
            }
        return new_det

    #Returns the traffic light state red 0 green 1 
    def _get_light_state(self, det):
        light_box = self.image[det['y1']:det['y2'], det['x1']:det['x2'], :]

        #HSV threshold
        thresholded_box = []

        #If debug is True thresholding is done inside the trackbar
        if self.debug is True:
            for bar in self.light_bars:
                thresholded_box.append(bar.in_range(light_box, show=self.debug))
        else: 
            hsv_image = cv2.cvtColor(light_box, cv2.COLOR_RGB2HSV)
            for i in range(len(self.initial_traffic_light_ranges)):
                threshold_image = cv2.inRange(hsv_image, 
                                                self.initial_traffic_light_ranges[i][0],
                                                self.initial_traffic_light_ranges[i][1])
                thresholded_box.append(threshold_image)

        red_count = np.count_nonzero(
            cv2.bitwise_or(thresholded_box[0], thresholded_box[1])
        )
        green_count = np.count_nonzero(thresholded_box[2])

        if red_count >= green_count:
            return int(0)
        else:
            return int(1)

    #Returns the traffic sign stop 2 other 3
    def _get_sign(self, det):
        #Calculate traffic sign aspect ratio
        box_aspect_ratio = float(abs(det['y1'] - det['y2'])) / float(abs(det['x1'] - det['x2']))
        if det['conf'] > 0.8 and box_aspect_ratio < 1.2 and box_aspect_ratio > 0.8:
            sign_box = self.image[det['y1']:det['y2'], det['x1']:det['x2'], :]

            #HSV threshold
            thresholded_box = []
            #If debug is True thresholding is done inside the trackbar
            if self.debug is True:
                for bar in self.sign_bars:
                    thresholded_box.append(bar.in_range(sign_box, show=self.debug))
            else: 
                hsv_image = cv2.cvtColor(sign_box, cv2.COLOR_RGB2HSV)
                for i in range(len(self.initial_traffic_sign_ranges)):
                    threshold_image = cv2.inRange(hsv_image, 
                                                    self.initial_traffic_sign_ranges[i][0],
                                                    self.initial_traffic_sign_ranges[i][1])
                    thresholded_box.append(threshold_image)

            stop_count = np.count_nonzero(thresholded_box[0])
            other_count = np.count_nonzero(thresholded_box[1])

            if stop_count >= other_count:
                return int(2)
            else:
                return int(3)

        else:
            return int(3)

    #Returns the image with bounding boxes drawn
    def draw_bboxes(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_det = []
        self.image_det = copy.deepcopy(self.image)
        if len(self.bboxes) != 0:
            for bbox in self.bboxes:
                self.image_det = cv2.rectangle(self.image_det,
                                    (bbox['x1'], bbox['y1']),
                                    (bbox['x2'], bbox['y2']),
                                    (0, 0, 255),
                                    2)
                cv2.putText(self.image_det,
                            self.class_names[bbox['c']],
                            (bbox['x1'], bbox['y1'] - 10),
                            font,
                            1,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA)
        return self.image_det

    #Get detections image
    def get_detections_image(self):
        return self.image_det

    def __str__(self):
        return "YOLO traffic light and sign detector\n" + str(self.class_names)