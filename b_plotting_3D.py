import os, sys
import pandas as pd
from icecream import ic
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

column_labels = ["x","y","z","bx","by","bz"]

#dat = pd.read_csv("test_b_out.fld",skiprows=[0,1],header=None,delim_whitespace=True)
#dat = pd.read_csv("1cm_test1_out.fld",skiprows=[0,1],header=None,delim_whitespace=True)
dat = pd.read_csv("proper_current_1mm.fld",skiprows=[0,1],header=None,delim_whitespace=True)

dat.columns = ["x","y","z","bx","by","bz"]


#plane = (dat.query("z==0"))
plane0 = dat.iloc[::2]
plane = plane0.query("z>0 & bz <0")
ic(plane)
# plane.to_csv("plane.csv")
# print(plane)
# sys.exit()
X = plane["x"].to_numpy()
Y = plane["y"].to_numpy()
Z = plane["bz"].to_numpy()

print(Z.max())
#sys.exit()
X_u = np.unique(X)
ic(X_u)
Y_u = np.unique(Y)
ic(Y_u)
#X_2D = np.reshape(X,(-1,int(np.sqrt(len(Z)))))
#Y_2D = np.reshape(Y,(-1,int(np.sqrt(len(Z)))))
#Z_2D = np.reshape(Z,(-1,int(np.sqrt(len(Z)))))

# ic(X_2D)
# #sys.exit()



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Grab some test data.
# X_u, Y_u, Z_u = axes3d.get_test_data(0.05)
# ic(X_u)
# ic(Y_u)
# ic(Z_u)


# # Plot a basic wireframe.
# #ax.plot_wireframe(X_u, Y_u, Z_2D, rstride=10, cstride=10)
# #ax.plot_surface(X_u,Y_u,Z_u)
# ax.plot_surface(X_2D,Y_2D,Z_2D,cmap=cm.coolwarm)
# ax.set_xlabel('$X$', fontsize=20)
# ax.set_ylabel('$Y$')
# ax.zaxis.set_rotate_label(False) 
# ax.set_zlabel('$B_z$', fontsize=30, rotation = 0)

# plt.show()

#cm = plt.get_cmap("RdYlGn")

#x = np.random.rand(30)
#y = np.random.rand(30)
#z = np.random.rand(30)
#col = np.arange(30)

# x = X
# y = Y
# z = plane["z"].to_numpy()
# col = Z

# # 2D Plot
# fig = plt.figure()
# #ax = fig.add_subplot(111)
# #ax.scatter(x, y, s=10, c=col, marker='o')  

# # 3D Plot
# fig = plt.figure()
# ax3D = fig.add_subplot(111, projection='3d')
# p3d = ax3D.scatter(x, y, z, s=30, c=col, marker='o',alpha=0.5, antialiased=True, zorder = 0.5)           
# plt.show()

# x_list = [1,2,3,4]
# y_list = [1,2,3,4]
# z_list = [1,2,3,4]
# i_list = [1,2,3,4]

x_list = X
y_list = Y
z_list = plane["z"].to_numpy()
i_list = np.sqrt(np.square(plane["bz"].to_numpy()))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# choose your colormap
cmap = plt.cm.jet

# get a Nx4 array of RGBA corresponding to zs
# cmap expects values between 0 and 1
#z_list = np.array(z_list) # if z_list is type `list`
ak = np.sqrt(np.square(i_list)) / np.sqrt(np.square(i_list)).max()
print(ak.max())
print(ak.min())

colors = cmap(ak)
ic(colors)

# set the alpha values according to i_list
# must satisfy 0 <= i <= 1
i_list = np.array(i_list)
colors[:,-1] = i_list / i_list.max()
ic(colors)

print(colors)

# then plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_list, y_list, z_list, c=colors)
ax.set_xlabel('$X$', fontsize=20)
ax.set_ylabel('$Y$', fontsize=20)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$Z$', fontsize=20, rotation = 0)
plt.show()

#https://stackoverflow.com/questions/30986848/how-can-i-control-the-color-and-opacity-of-each-point-in-a-python-scatter-plot
#https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
#https://matplotlib.org/2.0.2/examples/pylab_examples/custom_cmap.html