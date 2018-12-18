# iptrack - interpolate track
#
# SYNTAX
# p=iptrack(filename)
#
# INPUT
# filename: data file containing exported tracking data on the standard
# Tracker export format
#
# mass_A
# t	x	y
# 0.0	-1.0686477620876644	42.80071293284619
# 0.04	-0.714777136706708	42.62727536827738
# ...
#
# OUTPUT
# p=iptrack(filename) returns the coefficients of a polynomial of degree 15
# that is the least square fit to the data y(x). Coefficients are given in
# descending powers.

import os
import numpy as np
import matplotlib.pyplot as plt
import re

def iptrack(filename):
    data = np.loadtxt(filename,skiprows=2)
    return np.polyfit(data[:,0],data[:,3],15)

def plot(filename1, filename2, filename3):
    data_1 = np.loadtxt(filename1, skiprows=2)
    x1_coordinates = data_1[:,0]
    y1_coordinates = data_1[:,3]

    data_2 = np.loadtxt(filename2, skiprows=2)
    x2_coordinates = data_2[:, 0]
    y2_coordinates = data_2[:, 3]

    data_3 = np.loadtxt(filename3, skiprows=2)
    x3_coordinates = data_3[:, 0]
    y3_coordinates = data_3[:, 3]

    trackID_1 = 'Baneprofil ' + findTrackID(filename1)
    trackID_2 = 'Baneprofil ' + findTrackID(filename2)
    trackID_3 = 'Baneprofil ' + findTrackID(filename3)

    plt.plot(x1_coordinates-0.04, y1_coordinates, x2_coordinates-0.11, y2_coordinates, x3_coordinates-51.29, y3_coordinates)
    #plt.title("Baneprofiler")
    plt.ylabel('$v(t)$ [m/s]')
    plt.xlabel('$t$ [s]')
    plt.legend((trackID_1, trackID_2, trackID_3), shadow=True, loc=(.05, 0.8))
    plt.grid()
    plt.show()

def findTrackID(filename):
	return re.search(r'\d+',filename).group()

data1 = "../baneprofil1fart.txt"
data2 = "../baneprofil2fart.txt"
data3 = "../baneprofil3fart.txt"


read1 = open(data1, "r")
read2 = open(data2, "r")
read3 = open(data3, "r")


baneprofil1 = iptrack(read1)
print(baneprofil1)
baneprofil2 = iptrack(read2)
print(baneprofil2)
baneprofil3 = iptrack(read3)
print(baneprofil3)

plot(data1, data2, data3)

