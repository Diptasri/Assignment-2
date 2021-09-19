
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import pi
import math

V = np.array(([9,0],[0,4]))
u = np.array(([0,0]))
p = np.array(([0,1]))
f = -36
a=3
b=2
V_inv = np.linalg.inv(V)
C = np.vstack((V_inv * u))
lambda_1 = a*a
lambda_2 = b*b
e = math.sqrt (1 - lambda_2/lambda_1)
l = math.sqrt (lambda_1)
n= l * p
D = (0, [a/e],[-a/e])
F = ([0, a*e , -a*e])

t = np.linspace(0, 2*pi, 100)
plt.plot( C[0]+a*np.sin(t) , C[1]+b*np.cos(t), color= 'b', label='Ellipse')
plt.scatter(C[0], C[1] , color= 'b', marker = 'o',label='Center')
plt.axhline(y=D[1], color= 'r')
plt.axhline(y=D[2], color= 'r')
plt.plot(D[0] , D[1] , 'r', label='Directrix')
plt.plot(D[0] , D[2] , 'r', )
plt.plot(F[0] , F[1] , 'go', label='Foci')
plt.plot(F[0] , F[2] , 'go' )

plt.text(C[0] * (1 - 0.1), C[1] * (1 - 0.2) , '(0,0)')
plt.text(D[0], D[1] , 'y= 9/√5')
plt.text(D[0], D[2] , 'y= -9/√5')
plt.text(F[0] * (1 - 0.1), F[1] * (1 - 0.2) , '(0, √5/3)')
plt.text(F[0] * (1 - 0.1), F[2] * (1 - 0.2) , '(0, -√5/3)')

plt.title("Plot of Ellipse")
plt.xlabel("X- Axis")
plt.ylabel("Y- Axis")
plt.grid(True)
plt.legend(loc='best')
plt.show()