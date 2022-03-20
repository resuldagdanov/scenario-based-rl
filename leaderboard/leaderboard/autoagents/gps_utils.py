import numpy as np 
from scipy.spatial import distance  
import math


def interpolation(coords, ds = 1.5):
    _coords = np.array(coords)

    x_coords =_coords[:, 0]
    y_coords = _coords[:, 1]

    dense_coords = []

    for i in range(len(coords) - 1):
        # checking distance between each points
        dist = distance.euclidean(_coords[i], _coords[i + 1])
        if (dist > ds):
            # adding number of required data points for interpolation
            ratio = dist / float(ds) 
            x_step = (x_coords[i + 1] - x_coords[i]) / ratio
            y_step = (y_coords[i + 1] - y_coords[i]) / ratio

            for j in range(int(round(ratio))):
                dense_coords.append([x_coords[i] + (j * x_step), y_coords[i] + (j * y_step)])

    _dense_coords = np.array(dense_coords)
    return _dense_coords


def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters

    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2-f)
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z         
        
def ecef_to_enu(x, y, z, lat0, lon0, h0):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2-f)

    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp
