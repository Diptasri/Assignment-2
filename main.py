import numpy as np
from matplotlib import pyplot as plt
from math import pi

x=0     #x-position of the center
y=0     #y-position of the center
a=2     #radius on the x-axis
b=3     #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
plt.plot( x+a*np.sin(t) , y+b*np.cos(t) )
plt.scatter(x, y , color= 'r', marker = 'o')
plt.text(x, y + 0.5, '({},{})'.format(x, y))
plt.title("Plot of Ellipse")
plt.xlabel("X- Axis")
plt.ylabel("Y- Axis")
plt.grid(True)
plt.show()
