
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





















'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
def parab_gen(y):
    x=y**2
    return x
#parab parameters
V = np.array(([0,0],[0,1]))
u = np.array(([-1/2,0]))
f = 0
p = np.array(([1,0]))

eta = np.array([p@u])
a=-0.5*p-u
b=u+eta*p
c1 = np.vstack((u+eta*p,V))
c2 = np.vstack((-f,(a).reshape(-1,1)))
c = LA.lstsq(c1,c2,rcond=None)[0]
c = c.flatten()
print(c)

n = np.array(([1,0]))
lambda_2=1
e=1
D=np.array(([(u@u-lambda_2*f)/(2*e*e*u@n),0]))
print(D)
F=-D
print(F)
y=np.linspace(-1,1,1000)
x=y**2
plt.text(c[0] * (1 - 0.1), c[1] * (1 - 0.2) , '(0,0)')
plt.text(F[0] * (1 - 0.1), F[1] * (1 - 0.2) , '(1/4,0)')
plt.text(D[0] * (1 - 0.1), D[1] * (1 - 0.2) , 'x=-1/4')
plt.plot(0,0,'ro',label='Vertex point')
plt.plot(1/4,0,'go',label='Focus point')
plt.plot(-1/4,0,label='Directrix')
plt.axvline(x=-0.25, color ="blue")
plt.plot(x,y)
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid()
plt.legend(loc='best')
plt.show()
'''
'''
import numpy as np
from matplotlib import pyplot as plt
from math import pi

C = np.array([0 , 0])    #Center Point
F = np.array([0, 0.75, -0.75])  #Foci Point
D = np.array([0 , 4.02 , -4.02]) #Directrix Point
a=2    #semi minor axis
b=3    #semi major axis

t = np.linspace(0, 2*pi, 100)
plt.plot( C[0]+a*np.sin(t) , C[1]+b*np.cos(t), color= 'b', label='Ellipse')
plt.scatter(C[0], C[1] , color= 'b', marker = 'o',label='Center')
plt.axhline(y=4.02, color= 'r')
plt.axhline(y=-4.02, color= 'r')
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
'''