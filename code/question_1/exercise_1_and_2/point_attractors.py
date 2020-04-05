import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

"""Script to create memory in state space plots as in Figure 2"""

# create meshgrid
x = np.linspace(0, 100, 1000)
y = np.linspace(0, 100, 1000)
X, Y = np.meshgrid(x, y)

# FUNCTIONS FOR PLOTS IN FIGURE 2
# function for right plot
f = np.sin(0.2*X) + 3*np.cos(0.3*X) + 2*np.sin(0.2*Y) + 2*np.cos(0.4*Y) 

# function for right plot
#f = np.sin(0.15*X) + np.cos(0.15*Y)  
#f = f.clip(max=0)

# create contour plot
h = plt.contourf(X,Y,f, cmap='gray')
# no axes
plt.axis('off')

plt.show()