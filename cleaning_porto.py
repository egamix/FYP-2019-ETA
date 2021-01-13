from __future__ import print_function

from IPython import display

import math
import matplotlib
import sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import radians, cos, sin, arcsin, arccos, sqrt, pi, arctan2, degrees, arctan
import itertools
from datetime import datetime
from scipy import signal,ndimage, misc, stats
 
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()
tqdm.pandas(tqdm_notebook)

import osrm
from joblib import dump, load
import ast

def haversine(lat1, lon1, lat2, lon2):
    #ensure using numpy and not math, or pandas series cannot be passed
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a))
    r = 6378.137 ##radius of earth km
    return c * r

def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def cosrule_arr(arr):
    r = 6378.137
    next_arr = shift5(arr,1)
    c = haversine(arr[:,0], arr[:,1],
                    next_arr[:,0], next_arr[:,1])
    next_arr2 = shift5(next_arr, 1)
    a = haversine(next_arr[:,0], next_arr[:,1],
                    next_arr2[:,0], next_arr2[:,1])
    b = haversine(arr[:,0], arr[:,1],
                    next_arr2[:,0], next_arr2[:,1])
    a = a/r
    b= b/r
    c = c/r
    cosb = (c**2 + a**2 - b**2)/(2*a*c)
    return np.arccos(cosb) * 57.2958

def cosrule(df):
    r = 6378.137
    next_df = df.shift(1)
    c = haversine(df.iloc[:,0], df.iloc[:,1],
                    next_df.iloc[:,0], next_df.iloc[:,1])
    next_df2 = next_df.shift(1)
    a = haversine(next_df.iloc[:,0], next_df.iloc[:,1],
                    next_df2.iloc[:,0], next_df2.iloc[:,1])
    b = haversine(df.iloc[:,0], df.iloc[:,1],
                    next_df2.iloc[:,0], next_df2.iloc[:,1])
    a = a/r
    b= b/r
    c = c/r
    cosb = (c**2 + a**2 - b**2)/(2*a*c)
    return np.arccos(cosb) * 57.2958
def spherical_dist(pos1, pos2, r=6378137):
    '''
    stackexchange 19413259
    '''
    pos1 = pos1 * pi/180
    pos2 = pos2 * pi/180
    cos_lat1 = cos(pos1[..., 1])
    cos_lat2 = cos(pos2[..., 1])
    cos_lat_d = cos(pos1[..., 1] - pos2[..., 1])
    cos_lon_d = cos(pos1[..., 0] - pos2[..., 0])
    return r * arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def time_dif(time1, time2):
    return abs(time1 - time2) / np.timedelta64(1, 's')

def speed_array_calc(df, N, window_size):
    '''
    calculates the distance, time and speed of every point wrt to every other point
    '''
    locations = df[["longitude", "latitude"]].iloc[N:N+window_size].values.astype(float)
    dist_array = spherical_dist(locations[:, None], locations)
    try:
        time_val = np.array(pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise').iloc[N:N+window_size])
    except(KeyError):
        time_val = np.array(pd.to_datetime(df["iso timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise').iloc[N:N+window_size])
#     time_val = np.array(pd.to_datetime(kinematics["time"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise'))
    time_array = time_dif(time_val[:, None], time_val)
    speed_array = np.divide(dist_array, time_array)#, out = np.zeros_like(dist_array), where=time_array!=0)
    '''
    calculate z score of the array
    ideally a point very far off from the rest would have high zscore for its speed
    '''
    speed_array = np.ma.array(speed_array, mask = np.isnan(speed_array))
    speed_array = abs(stats.zscore(speed_array, axis=None))
    np.fill_diagonal(speed_array, 0)
    speed_array[speed_array < 1.96] = 0
    speed_array[speed_array > 0 ] = 1
    # # speed_array = scipy.special.expit(speed_array)
    # np.fill_diagonal(speed_array, 0)
    return speed_array

def speed_array_calc_mad(df, N, window_size):
    np.set_printoptions(precision=3)
    locations = df[["longitude", "latitude"]].iloc[N:N+window_size].values.astype(float)
    dist_array = spherical_dist(locations[:, None], locations)
#     print(dist_array)
    time_val = np.array(pd.to_datetime(df["iso timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise').iloc[N:N+window_size])
#     time_val = np.array(pd.to_datetime(kinematics["time"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise'))
    time_array = time_dif(time_val[:, None], time_val)
    speed_array = np.divide(dist_array, time_array)#, out = np.zeros_like(dist_array), where=time_array!=0)
    # print(speed_array)

    speed_array = np.ma.array(speed_array, mask = np.isnan(speed_array))

    speed_array = mad_based_outlier(speed_array)
    np.fill_diagonal(speed_array, 0)
    # print(speed_array)
    return speed_array

def mad_based_outlier(points, thresh=3.5):
    shp = points.shape
    points = points.flatten()
    if len(points.shape) ==1:
        points = points[:,None]
    median = np.ma.median(points)
    diff = np.ma.sum((points-median)**2, axis = -1)
    diff = np.ma.sqrt(diff)
    med_abs_deviation = np.ma.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    points = modified_z_score > thresh
    points = np.reshape(points, shp)
    return 1*points

def outlier_find(df, window_size):
    '''
    Based on Koyak algorithm for outlier detection
    Generates the list 'out' of indexs which are deemed outliers wrt to other points
    the input A array is still experimental
    '''
    out = np.zeros(0)
    for N in range(0, df.shape[0], window_size):
        A = speed_array_calc_mad(df, N, window_size)
        n = A.shape[0]
        b = np.sum(A, axis = 1)
        o = np.zeros(shape = n)
        while ((np.amax(b) > 0)):
            r = np.argmax(b)
            o[r] = 1
            b[r] = 0
            for j in range (0, n):
                if (o[j] == 0):
                    b[j] = b[j] - A[r][j]
        out = np.append(out, o)
    return out

def compute_dist(df):
    next_df = df.shift(1)
    dist = haversine(df.iloc[:,0], df.iloc[:,1],
                    next_df.iloc[:,0], next_df.iloc[:,1])
    return dist

def compute_time(df):
    next_df = df.shift(1)
#     df["time"] = pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S.%f", errors='raise')
    timedelt = df["iso timestamp"] - next_df["iso timestamp"]
    return timedelt

def compute_speed(df):
    kinematics = df.copy()
    kinematics["distance_travelled"] = compute_dist(kinematics[["latitude", "longitude"]].astype(float)).values
    kinematics["time_elapsed"] = compute_time(kinematics).values
    kinematics["time_elapsed_seconds"] = kinematics["time_elapsed"]/np.timedelta64(1,'s')
    kinematics["speed m/s"] = (kinematics["distance_travelled"]*1000)/kinematics["time_elapsed_seconds"]
    kinematics["speed kmh"] = kinematics["speed m/s"]*3.6
    kinematics.drop(columns = ['time_elapsed'], inplace = True)
    kinematics.fillna(0, inplace = True)
#     df["distance_travelled"] = kinematics["distance_travelled"].values
#     df["speed kmh"] = kinematics["speed kmh"].values
    return kinematics

def cal_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two points using the formula
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    """
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y1 = cos(lat1) * sin(lat2)
    y2 = sin(lat1) * cos(lat2) * cos(dlon)
    y = y1 - y2

    initial_bearing = arctan2(x, y)

    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing  

# chunk = 0
# taxi = []
# for df in pd.read_csv('/mnt/hgfs/FYP/train_porto.csv', chunksize = 100000):
#     taxi.append(df)
#     chunk += 1
#     if chunk > 2:
#         break

chunk = 0

for df in  pd.read_csv('/mnt/hgfs/FYP/train_porto.csv', chunksize = 2):
    print('read')

    out = []

    for index, row in tqdm(df.iterrows()):
        try: 
            time_stamp = row['TIMESTAMP']
            polyline = ast.literal_eval(row['POLYLINE'])
            ts = []
            for i in range(0,len(polyline)):
                ts.append(pd.to_datetime(time_stamp, unit = 's'))
                time_stamp += 15

            foo = np.append(np.array(ts).reshape(-1,1), polyline, axis = 1)  
            foo = pd.DataFrame(foo)
            foo.columns = ['time', 'longitude', 'latitude']

            wz = int(30 * arctan(0.005*foo.shape[0]))

            if (wz > 5):
                outliers = outlier_find(foo, window_size=wz).astype(bool)
            else:
                outliers = np.zeros(foo.shape[0]).astype(bool)

            foo = foo[~outliers]
            #         polyline = ast.literal_eval(row['POLYLINE'])
            a = np.array(foo[['time', 'longitude', 'latitude']].values)
            ang = cosrule_arr(a[:,1:].astype(float))
            a = np.append(a, ang.reshape(-1,1), axis = 1)
            # a = np.append(a, np.array(ts).reshape(-1,1), axis =1)
            while (a[:,3] < 30).any():
                a = a[~shift5(a[:,3] < 30, -1)] 
                a[:,3] = cosrule_arr(a[:,1:3].astype(float))
        except IndexError:
            pass

        a = pd.DataFrame(a[:,0:3])
        a.columns = ['time', 'longitude', 'latitude']
        a['ID'] = row['TAXI_ID']

        # print(len(polyline) - len(a))
        out.append(a)

    out2 = pd.concat(out)
    print(out2)

    break
    # out2.to_csv('./all_cleaned.tsv')