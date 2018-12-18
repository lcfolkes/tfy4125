# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

def iptrack (filnavn):
	data=np.loadtxt(filnavn,skiprows=2)
	return np.polyfit(data[:,1],data[:,2],15)
	
def trvalues(p,x):
	y=np.polyval(p,x)
	dp=np.polyder(p)
	dydx=np.polyval(dp,x)
	ddp=np.polyder(dp)
	d2ydx2=np.polyval(ddp,x)
	alpha=np.arctan(-dydx)
	R=(1.0+dydx**2)**1.5/d2ydx2
	return [y,dydx,d2ydx2,alpha,R]

baneprofil1 = "../baneprofil1fart.txt"
baneprofil2 = "../baneprofil2fart.txt"
baneprofil3 = "../baneprofil3fart.txt"

polynomprofil1 = iptrack(baneprofil1)
polynomprofil2 = iptrack(baneprofil2)
polynomprofil3 = iptrack(baneprofil3)

time_1 = list(np.loadtxt(baneprofil1, skiprows=2, usecols=0))
time_1_2 = list(np.loadtxt(baneprofil2, skiprows=2, usecols=0))
time_3 = list(np.loadtxt(baneprofil3, skiprows=2, usecols=0))

listx = []
listy1 = []
listy2 = []
listy3 = []

g = 9.81 #Gravitasjonskonstanten
c = 2/5 #Neoprenkule (usikkerhet)

for i in range (1200):
    j = i * 0.001
    listx.append(j)

    y1, dydx1, d2ydx21, alpha1, R = trvalues(polynomprofil1, j)
    a1 = (g*np.math.sin(alpha1)) / (1+c)
    listy1.append(y1)

    y2, dydx2, d2ydx22, alpha2, R = trvalues(polynomprofil2, j)
    a2 = (g*np.math.sin(alpha2)) / (1+c)
    listy2.append(y2)

    y3, dydx3, d2ydx23, alpha3, R = trvalues(polynomprofil3, j)
    a3 = (g*np.math.sin(alpha3)) / (1+c)
    listy3.append(y3)

# print(time_1)
# print(listy1)
# plt.figure()
# plt.plot(listx,listy1)  # plotting the position vs. time: x(t)
# plt.plot(listx,listy2)
# plt.plot(listx,listy3)
# plt.title("Baneprofiler")
# plt.ylabel('$y(x)$ [m]')
# plt.xlabel('$x$ [m]')
# plt.legend(("Baneprofil 1", "Baneprofil 2", "Baneprofil 3"), shadow=True, loc=(.65, .7))
# plt.grid()
# plt.show()

#Euler's metode kun ta inn alpha
def euler(poly):
    N = 900  # Antall steg
    h = 0.001  # Steglengden
    t = np.zeros(N + 1)
    x = np.zeros(N + 1)
    v = np.zeros(N + 1)
    normal = np.zeros(N + 1)
    f = np.zeros(N + 1)
    t[0] = 0
    x[0] = 0
    v[0] = 0
    f[0] = 0
    normal[0] = 0
    resultattid = 0
    for n in range(N):
        alpha = trvalues(poly, x[n])[3]
        R = trvalues(poly, x[n])[4]
        x[n+1] = x[n] + h * (v[n] * np.math.cos(alpha))
        v[n+1] = v[n] + h * (g * np.math.sin(alpha))/(1 + c)
        t[n+1] = t[n] + h
        normal[n+1] = math.pow(v[n],2)/R + g*np.math.cos(alpha)
        f[n+1] = c*(g * np.math.sin(alpha))/(1 + c)

        #print(x[n])
        if resultattid == 0:
            if x[n+1] >= 1.15:
                resultattid = t[n+1]
                break
    t = t[:n]
    x = x[:n]
    v = v[:n]
    f = f[:n]
    normal = normal[:n]
    return t, x, v, normal, f


t1,x1,v1,normal1,f1 = euler(polynomprofil1)
t2,x2,v2,normal2,f2 = euler(polynomprofil2)
t3,x3,v3,normal3,f3 = euler(polynomprofil3)

plt.figure()
plt.plot(t1,f1)  # plotting the position vs. time: x(t)
plt.plot(t2,f2)
plt.plot(t3,f3)
#plt.title("Normalkraft")
plt.ylabel('$f/m$ $[\mathrm{m}\mathrm{s}^{-2}]$')
plt.xlabel('$t$ [s]')
plt.legend(("Baneprofil 1", "Baneprofil 2", "Baneprofil 3"), shadow=True, loc=(0.6, 0.65))
plt.grid()
plt.show()