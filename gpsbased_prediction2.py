import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import utm
from scipy.interpolate import UnivariateSpline

def compute_distance(x, y):
    """Compute traveled distance along the path.  x, y points are given as numpy arrays"""
    points = np.vstack((x, y)).T
    x_s = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    x_s = np.insert(x_s, 0, 0)
    return x_s

def compute_curvature(sensor_lat, sensor_lon, gps_lat, gps_lon, vel, t, n, pred_time):
    """sensor_lat, sensor_lon: gps sensor lat and lon
       gps_lat, gps_lon: map lat and lon
        vel: velocity
        t: unitless time (increasing array used in curvature calc)
        n: number of points around the center point to be used in curvature calc
        pred_time: prediction time"""
    ####
    sensorgps_lat =[]
    sensorgps_lon = []
    for jj in range(len(sensor_lat)):
        p0 = np.array([sensor_lat[jj], sensor_lon[jj]])
        gps_dist = np.sqrt(np.power(p0[0] - gps_lat, 2) + np.power(p0[1] - gps_lon, 2))
        idx = np.argpartition(gps_dist, 2)[:2]
        p1 = np.array([gps_lat[idx[0]], gps_lon[idx[0]]])
        p2 = np.array([gps_lat[idx[0]-1], gps_lon[idx[0]-1]])
        p3 = np.array([gps_lat[idx[0]+1], gps_lon[idx[0]+1]])

        v = p0 - p1
        s2 = p2 - p1
        s3 = p3 - p1

        if np.dot(s2,v) > 0:
            s = s2
        else:
            s = s3

        pop = p1 + np.dot(s,v)/np.dot(s,s) * s

        sensorgps_lat.append(pop[0])
        sensorgps_lon.append(pop[1])

    plt.figure(3)
    plt.plot(sensorgps_lat, sensorgps_lon, 'r.')
    plt.plot(sensor_lat, sensor_lon, 'g.')
    plt.plot(gps_lat, gps_lon, 'b.')
    #plt.show()
    #################################################
    utm_point = utm.from_latlon(np.array(sensorgps_lat), np.array(sensorgps_lon))
    x_map = utm_point[0]
    y_map = utm_point[1]
    x_s_map = compute_distance(x_map, y_map)

    dx_list = []
    dy_list = []
    ddx_list = []
    ddy_list = []
    cur_vel = []
    for j in range(100, len(sensor_lat)-100):
        target_dist = vel[j] * pred_time
        mid = j
        while mid==j or cur_dist<target_dist:
            mid+=1
            cur_dist = x_s_map[mid] - x_s_map[j]

        tpoints = t[mid-int(n/2) : mid+int(n/2)]
        xpoints = x_map[mid-int(n/2):mid+int(n/2)]
        ypoints = y_map[mid- int(n/2): mid + int(n/2)]

        features = np.array([np.ones((len(tpoints))), np.array(tpoints), np.array(tpoints) ** 2]).T

        x_estimate = np.dot(np.dot(np.linalg.inv(np.dot(features.T, features)), features.T), np.array(xpoints))
        dx = x_estimate[1] + 2 * x_estimate[2] * t[mid]
        ddx = 2*x_estimate[2]
        dx_list.append(dx)
        ddx_list.append(ddx)

        y_estimate = np.dot(np.dot(np.linalg.inv(np.dot(features.T, features)), features.T), np.array(ypoints))
        dy = y_estimate[1] + 2 * y_estimate[2] * t[mid]
        ddy = 2*y_estimate[2]
        dy_list.append(dy)
        ddy_list.append(ddy)

        cur_vel.append(vel[j])

    kappa = (np.array(dx_list) * np.array(ddy_list) - np.array(dy_list) * np.array(ddx_list)) / np.power(np.array(dx_list) ** 2 + np.array(dy_list) ** 2, 1.5)
    ay_kin = np.array(cur_vel) ** 2 * kappa

    return kappa, ay_kin

plt.close('all')
sensor_df = pd.read_csv('35halfmoon_SensorData.csv')
map_df = pd.read_csv('35halfmoon_GoogleMap.csv')

lat_list = []
lon_list = []
vel_x_list = []
vel_y_list = []
time_list = []
ax_list = []
ay_list = []
az_list =[]
index = []
time0 = 0
ii = 0
for j in range(len(sensor_df)):
    lat = sensor_df['fix__latitude'].iloc[j]
    lon = sensor_df['fix__longitude'].iloc[j]
    if ~np.isnan(lat) and ~np.isnan(lat):
        k = j
        while np.isnan(sensor_df['vel__twist_linear_x'].iloc[k]) and j-k<7:
            k-=1

        m = j
        while np.isnan(sensor_df['imu_data__accel_linear_y'].iloc[m]) and j-m<7:
            m-=1

        l = j
        while np.isnan(sensor_df['time_reference__time_ref_nsecs'].iloc[l]) and l-j<3:
            l+=1

        if j-k < 7 and l-j<3:
            lat_list.append(lat)
            lon_list.append(lon)
            vel_x_list.append(sensor_df['vel__twist_linear_x'].iloc[k])
            vel_y_list.append(sensor_df['vel__twist_linear_y'].iloc[k])
            ax_list.append(sensor_df['imu_data__accel_linear_x'].iloc[m])
            ay_list.append(sensor_df['imu_data__accel_linear_y'].iloc[m])
            az_list.append(sensor_df['imu_data__accel_linear_z'].iloc[m])
            time = sensor_df['time_reference__time_ref_secs'].iloc[l] + sensor_df['time_reference__time_ref_nsecs'].iloc[l]*1e-9
            if time0 == 0:
                time0 = time
            time -= time0
            time_list.append(time)
            index.append(ii)
            ii+=1

data_df = pd.DataFrame({'Time': time_list, 'Velx': vel_x_list, 'Vely': vel_y_list, 'lat': lat_list, 'lon':lon_list, 'Ax':ax_list, 'Ay':ay_list, 'Az':az_list})

vel = np.sqrt( np.array(data_df['Velx']) ** 2 + np.array(data_df['Vely']) ** 2 )

n = 20
pred_time = 2
kappa, ay_kin = compute_curvature(np.array(data_df['lat']), np.array(data_df['lon']), np.array(map_df['Lat']), np.array(map_df['Long']), vel, np.array(index), n, pred_time)

data_df['kappa'] = np.hstack( ( np.hstack( (np.zeros(100), kappa) ), np.zeros(100) ) )
data_df['ay_kin'] = np.hstack( ( np.hstack( (np.zeros(100), ay_kin) ), np.zeros(100) ) )

#data_df.to_csv('SensorCurvature_35halfmoon_4sec.csv')

plt.figure(1)
plt.plot(lat_list, lon_list)

ay =  data_df['Ay'][100:-100]
time = data_df['Time'][100:-100]

plt.figure(2)
plt.plot(time-pred_time,ay,'r', label='ay')
plt.plot(time, ay_kin*1000/9.8,'b', label='ay_kin')
plt.xlabel('Time')
plt.ylabel('Acceleration m/s^2')
plt.legend()
plt.title('Halfmoon.bag')
plt.show()



