import numpy as np
import cv2
from functools import partial

class RangeTrackbar:
    def __init__(self, name, space_type = 'HSV'):
        self.name = name

        self.max_value = 255
        self.space_type = space_type

        if space_type == 'HSV':
            self.space = {
                0: 'H',
                1: 'S',
                2: 'V'
            }
            self.max_first_value = 360//2
            self.space_range = np.array([[0, 0, 0,], [self.max_first_value, self.max_value, self.max_value]], dtype=np.int32)
        else:
            self.space = {
                0: 'R',
                1: 'G',
                2: 'B'
            }
            self.max_first_value = self.max_value
            self.space_range = np.array([[0, 0, 0,], [self.max_first_value, self.max_value, self.max_value]], dtype=np.int32)

        #Initial track_bar range

        cv2.namedWindow(self.name)

        for i in range(3):
            if self.space_type == 'HSV' and i == 0:
                cv2.createTrackbar(self.name + "low_" + self.space[i],  self.name , self.space_range[0][i], self.max_first_value, partial(self._track_bar_on_low, i))
                cv2.createTrackbar(self.name + "high_" + self.space[i], self.name , self.space_range[1][i], self.max_first_value, partial(self._track_bar_on_high, i))
            else:
                cv2.createTrackbar(self.name + "low_" + self.space[i],  self.name , self.space_range[0][i], self.max_value, partial(self._track_bar_on_low, i))
                cv2.createTrackbar(self.name + "high_" + self.space[i], self.name , self.space_range[1][i], self.max_value, partial(self._track_bar_on_high, i))

    def _track_bar_on_low(self, hsv, val):
        self.space_range[0][hsv] = val
        self.space_range[0][hsv] = min(self.space_range[1][hsv]-1, self.space_range[0][hsv])
        cv2.setTrackbarPos(self.name + "low_" + self.space[hsv], self.name, self.space_range[0][hsv])

    def _track_bar_on_high(self, hsv, val):
        self.space_range[1][hsv] = val
        self.space_range[1][hsv] = max(self.space_range[1][hsv], self.space_range[0][hsv]+1)
        cv2.setTrackbarPos(self.name + "high_" + self.space[hsv], self.name, self.space_range[1][hsv])

    #HSV value getter
    def get_range(self):
        return self.space_range

    #HSV value set list [[H0, S0, V0], [H1, S1, V1]]
    def set_range(self, update_range):
        self.space_range = np.array(update_range, dtype=np.int32)
        
        for i in range(3):
            cv2.setTrackbarPos(self.name + "low_" + self.space[i],  self.name , self.space_range[0][i])
            cv2.setTrackbarPos(self.name + "high_" + self.space[i], self.name , self.space_range[1][i])

    #Runs cv inrange and shows the image (BGR) in the box (optional)
    def in_range(self, image, show=True):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        threshold_image = cv2.inRange(hsv_image, self.space_range[0], self.space_range[1])

        if show is True:
            threshold_image_3 = np.concatenate((np.expand_dims(threshold_image, 2), 
                                                np.expand_dims(threshold_image, 2), 
                                                np.expand_dims(threshold_image, 2)), 
                                                axis=2)

            cv2.imshow(self.name, np.concatenate((image[:, :, ::-1], threshold_image_3), axis=1))
        
        return threshold_image

    def __str__(self):
        return ("Space: %s\n"%(self.space_type) + 
                "Low:  %d %d %d\n"%(self.space_range[0][0], self.space_range[0][1], self.space_range[0][2]) +
                "High: %d %d %d\n"%(self.space_range[1][0], self.space_range[1][1], self.space_range[1][2]))

if __name__ == '__main__':
    bar1 = RangeTrackbar("box1", space_type="HSV")
    while True:
        print(bar1)
        key = cv2.waitKey(33)
        if key == 'q':
            break