import os, sys
import pandas as pd
from icecream import ic
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

column_labels = ["x","y","z","bx","by","bz"]

#dat = pd.read_csv("test_b_out.fld",skiprows=[0,1],header=None,delim_whitespace=True)
dat = pd.read_csv("1cm_test1_out.fld",skiprows=[0,1],header=None,delim_whitespace=True)
dat.columns = ["x","y","z","bx","by","bz"]


plane = (dat.query("z==0"))
ic(plane)
# plane.to_csv("plane.csv")
# print(plane)
# sys.exit()
X = plane["x"].to_numpy()
Y = plane["y"].to_numpy()
Z = plane["bz"].to_numpy()

X_u = np.unique(X)
ic(X_u)
Y_u = np.unique(Y)
ic(Y_u)
X_2D = np.reshape(X,(-1,int(np.sqrt(len(Z)))))
Y_2D = np.reshape(Y,(-1,int(np.sqrt(len(Z)))))
Z_2D = np.reshape(Z,(-1,int(np.sqrt(len(Z)))))
ic(X_2D)
#sys.exit()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X_u, Y_u, Z_u = axes3d.get_test_data(0.05)
ic(X_u)
ic(Y_u)
ic(Z_u)


# Plot a basic wireframe.
#ax.plot_wireframe(X_u, Y_u, Z_2D, rstride=10, cstride=10)
#ax.plot_surface(X_u,Y_u,Z_u)
ax.plot_surface(X_2D,Y_2D,Z_2D,cmap=cm.coolwarm)
ax.set_xlabel('$X$', fontsize=20)
ax.set_ylabel('$Y$')
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$B_z$', fontsize=30, rotation = 0)

plt.show()
