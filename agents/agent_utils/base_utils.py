import carla
import numpy as np
np.random.seed(0)
from datetime import datetime

def get_time_info():
    today = datetime.today() # month - date - year
    now = datetime.now() # hours - minutes - seconds

    current_date = str(today.strftime("%b_%d_%Y"))
    current_time = str(now.strftime("%H_%M_%S"))

    # month_date_year-hour_minute_second
    time_info = current_date + "-" + current_time

    return time_info

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def get_angle_to(pos, theta, target):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle 

    return angle


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


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

        total = a + b
        threshold = 1.0
        if dist + threshold > total:
            continue

        result.append(light)

    return result

def tensorboard_writer(writer, eps, episode_total_reward, best_reward, policy_loss, value_loss, n_updates):
    writer.add_scalar("episode total reward - episode", episode_total_reward, eps)
    writer.add_scalar("best reward - episode", best_reward, eps)

    if n_updates != 0:
        writer.add_scalar("policy loss - episode", policy_loss / n_updates, eps)
        writer.add_scalar("value loss - episode", value_loss / n_updates, eps)
        
def tensorboard_writer_with_one_loss(writer, eps, episode_total_reward, best_reward, value_loss, n_updates):
    writer.add_scalar("episode total reward - episode", episode_total_reward, eps)
    writer.add_scalar("best reward - episode", best_reward, eps)

    if n_updates != 0:
        writer.add_scalar("value loss - episode", value_loss / n_updates, eps)

def tensorboard_writer_evaluation(writer, eps, episode_total_reward):
    writer.add_scalar("episode total reward - episode", episode_total_reward, eps)

def tensorboard_writer_running_average(writer, total_step_num, running_average):
    writer.add_scalar("running average (100 steps) - total step number", running_average, total_step_num)