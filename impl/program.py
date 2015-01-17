#!/usr/bin/python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

DV = [] #directionality vectors

#read directionality vectors from input file:
    #d11 d12 d13 d14 d15 d16 d17 d18
    #...
    #dN1 dN2 dN3 dN4 dN5 dN6 dN7 dN8
#N is directionality vectors count
with open('input.txt', 'r') as input:
    lines = input.readlines()

    for line in lines:
         d = []
         d = line.split(' ')
         d[len(d) - 1] = d[len(d) - 1].strip('\n')
         DV.append(d)

#calculate angular distance parameter between two vectors d and q:
    #return omegadq = 1 - sum(d1*q1 + ... + d8*q8)
def angular_distance_parameter(d, q):
    omega = 0
    for i in range(0, 8):
        omega = omega + float(d[i])*float(q[i])
    omega = 1 - omega
    return omega

#write angular distance parameters to log file:
    #d_index,q_index:omegadq
def write_log(omegadq, d_index, q_index):
    with open('log.txt', 'a') as log:
        log.write('d' + str(d_index + 1) + ',d' + str(q_index + 1)+ ':' 
                  + str(omegadq) + '\n')
    return

#clear log file
def clear_log():
    with open('log.txt', 'w') as log:
        log.write('')
    return

#plot angular distance parameters:
    #(x, y, z) - angular distance parameter z between frame x and frame y
    #centroid == 1 - plot centroid
def plot_angular_distance_parameters(X, Y, Z, centroid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    ax.scatter(X, Y, Z, c='b', marker='o')

    #TO DO: check centroid calculation
    if centroid == 1:
        centroidx = np.mean(X)
        centroidy = np.mean(Y)
        centroidz = np.mean(Z)
        
        ax.scatter(centroidx, centroidy, centroidz, c='r', marker='x')
 
    ax.set_xlabel('Frame x')
    ax.set_ylabel('Frame y')
    ax.set_zlabel('Angular distance parameter')
 
    plt.show()
    return





#TO DO: determine vectors for calculating angular distance parameters
    #for now as in the paper
omega = [] #angular distance parameter vector

#calculate angular distance parameter for (d1,d2), (d1, d3), ..., (d1,dN)
for i in range(1, len(DV)):
    omega.append(angular_distance_parameter(DV[0],DV[i]))

#write angular distance parameters to log file:
    #omega12
    #...
    #omega1N
clear_log()

for i in range(0, len(omega)):
    write_log(omega[i],0,i + 1)

#plot angular distance parameter for (d1,d2), (d1, d3), ..., (d1,dN)
#and it's centroid
X = []
Y = []
Z = []
for i in range(0, 1):
    for j in range(1, len(DV)):
        X.append(i + 1)
        Y.append(j + 1)
        
Z = omega

plot_angular_distance_parameters(X, Y, Z, 1)

print X
print Y
print Z