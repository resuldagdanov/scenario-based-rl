import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class DatasetLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subfolder_paths = []

        subfolders = ["rgb_front", "measurements"]

        for subfolder in subfolders:
            self.subfolder_paths.append(os.path.join(self.root_dir, subfolder))

        self.len = len(os.listdir(self.subfolder_paths[-1]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.subfolder_paths[0],  "%04i.png" % idx)
        meas_name = os.path.join(self.subfolder_paths[1],  "%04i.json" % idx)

        image = io.imread(img_name)
        image = np.array(image.transpose((2, 0, 1)), np.float32)
        
        # normalize image
        image = image / 255

        with open(meas_name, 'r') as f:
            meas_json = json.load(f) 

        # global gps position of the ego vehicle
        ego_x = meas_json['x']
        ego_y = meas_json['y']

        # far node
        x_command = meas_json['x_command']
        y_command = meas_json['y_command']

        # convert far nodes to relative local waypoints
        local_command_point = np.array([x_command - ego_x, y_command - ego_y], np.float32)

        sample = {
            "image": image,
            "velocity": np.array([meas_json['speed']], np.float32),
            "control": np.array([meas_json['throttle'], meas_json['steer'], meas_json['brake']], np.float32),
            "target_point": tuple(local_command_point),
            "speed_sequence": tuple(np.array(meas_json['speed_sequence'], np.float32)),
            "reward": np.array([meas_json['reward']], np.float32)
        }
        return sample
