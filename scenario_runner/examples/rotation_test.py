
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

def draw(ax, lights, car_location, car_rotation_d, lights_color, fov_d = 120):
    fov_wedge = Wedge(car_location, fov_radius, car_rotation_d-fov_d/2, car_rotation_d+fov_d/2)
    patches = [fov_wedge]
    patches.append(fov_wedge)
    p = PatchCollection(patches, alpha=0.4)
    colors = np.array([2])
    p.set_array(colors)
    ax.scatter(lights[:, 0], lights[:, 1], c=lights_color)
    ax.scatter(car_location[0], car_location[1], c='r')
    ax.add_collection(p)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.grid()
    ax.set_aspect('equal')

N = 20
lights=np.random.multivariate_normal([0, 0], cov=3*np.eye(2), size=N)
lights_color = []
for i in range(len(lights)):
    lights_color.append('b')
print(lights_color)
car_location = np.array([0.5 , 0.5])
car_rotation_d = 90
car_rotation = car_rotation_d / 180 * np.pi
fov_d = 120
fov = fov_d / 180 * np.pi
fov_radius = 2

fig, ax = plt.subplots(1, 3)
draw(ax[0], lights, car_location, car_rotation_d, lights_color, fov_d=fov_d)

car_centered = np.array([0, 0])
lights_centered = lights - car_location
draw(ax[1], lights_centered, car_centered, car_rotation_d, lights_color, fov_d=fov_d)

for i in range(len(lights_centered)):
    light_radius = np.linalg.norm(lights_centered[i])
    light_theta = np.arctan2(lights_centered[i, 1], lights_centered[i, 0]) / np.pi * 180 
    anglediff = (car_rotation_d - light_theta + 180 + 360) % 360 - 180   
    if light_radius < fov_radius:
        print(car_rotation_d, light_theta, anglediff) 
        if ((anglediff < fov_d/2 ) and (anglediff > -1 * fov_d/2)):
            lights_color[i] = 'g'


draw(ax[2], lights_centered, car_centered, car_rotation_d, lights_color, fov_d=fov_d)



plt.show()



