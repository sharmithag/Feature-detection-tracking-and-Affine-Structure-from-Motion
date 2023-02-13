import os
import random
import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
import scipy.linalg 
from mpl_toolkits.mplot3d import Axes3D
data = loadmat('/content/Part2_data/tracks.mat')
track_x = data['track_x']
track_y = data['track_y']

# Remove the nan value
# YOUR CODE HERE
track_x = track_x.T
track_y = track_y.T
nan_index = np.argwhere(np.isnan(track_x))

for i in range(51):
  
    x = track_x[i] 
    y = track_y[i]
    x = np.delete(x,nan_index)
    y = np.delete(y,nan_index)
  
    if i == 0:
      track_x_cln = x
      track_y_cln = y
    else:
      track_x_cln = np.vstack((track_x_cln,x))
      track_y_cln = np.vstack((track_y_cln,y))

def affineSFM(x, y):
  '''
  Function: Affine structure from motion algorithm
  % Normalize x, y to zero mean
  % Create measurement matrix
  D = [xn' ; yn'];
  % Decompose and enforce rank 3
  % Apply orthographic constraints
  '''
  # YOUR CODE HERE
   
  for i in range(51):
    x_cen = x[i]-np.mean(x[i])
    y_cen = y[i]-np.mean(y[i])
    
    if i == 0:
      D = np.vstack((x_cen,y_cen))
    else:
      D = np.vstack((D,x_cen,y_cen))
  U,W,VT = np.linalg.svd(D)
  W = np.diag(W)
  U3 = U[:,0:3]
  W3 = W[0:3,0:3]
  VT3 = VT[0:3,:]
  # Initial A and X
  M = np.dot(U3,np.sqrt(W3))
  S = np.dot(np.sqrt(W3),VT3)
  for i in range(0,102,2):
    temp = np.array([M[i][0]*M[i+1][0], M[i][1]*M[i+1][0],M[i][2]*M[i+1][0], M[i][0]*M[i+1][1],M[i][1]*M[i+1][1],M[i][2]*M[i+1][1],M[i][0]*M[i+1][2],M[i][1]*M[i+1][2],M[i][2]*M[i+1][2]])
    if i == 0:
      LS = temp
    else:
      LS = np.vstack((LS,temp))
  _,_,LT = np.linalg.svd(LS)
  L = reshape(LT[-1,:],(3,3))
  C = scipy.linalg.lu(L)
  # updated A an X
  A = np.dot(M,C[1])
  X = np.dot(np.linalg.inv(C[1]),S)
  return A,X

A,X = affineSFM(track_x_cln,track_y_cln)
# plotting X
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,3,1, projection='3d')
ax.scatter3D(X[0,:],X[1,:],X[2,:],marker = '.',color='r')
ax.view_init(90,60)
ax = fig.add_subplot(1,3,2, projection='3d')
ax.scatter3D(X[0,:],X[1,:],X[2,:],marker = '.',color='r')
ax.view_init(120,30)
ax = fig.add_subplot(1,3,3, projection='3d')
ax.scatter3D(X[0,:],X[1,:],X[2,:],marker = '.',color='r')
ax.view_init(90,30)
plt.show()

for i in range(0,102,2):
  Ak = np.cross(A[i,:],A[i+1,:])
  a = np.linalg.norm(Ak)
  temp = Ak/a
  if i == 0:
    Ak_norm = temp
  else:
    Ak_norm = np.vstack((Ak_norm,temp))
# plotting Normalized A
plt.rcParams["figure.figsize"] = (5,5)
plt.subplot(1,3,1)
plt.plot(Ak_norm[:,0])
plt.subplot(1,3,2)
plt.plot(Ak_norm[:,1])
plt.subplot(1,3,3)
plt.plot(Ak_norm[:,2])
plt.show()
