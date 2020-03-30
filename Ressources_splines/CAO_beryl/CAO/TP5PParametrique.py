# -*- coding: utf-8 -*-
# TP5parametricInterpolStudents.py
"""
Parametric interpolation of data acquired with the mouse with 
    uniform parameterization
    Chebyshev parameterization
    chordal parameterization
"""
import numpy as np
import matplotlib.pyplot as plt

def diffdiv(xi,yi) :
    """ Calculation of the vector delta of divided differences (DD)
        for interpolation data (xi,yi)
        Input : 
            xi, yi = two vectors of same size n+1 (n = degree)
        Output : 
            delta = vector of size 2n+1, precisely :
            delta(j) for j=0,1,...,n are the DD (first half)
            delta(j) for j=n+1,...,2n+1 are the other DD for updates
    """
    n = np.size(xi)
    delta = np.zeros(2*n-1)
    for i in range(n) :
        delta[2*i] = yi[i]
    for k in range(1,n) :
        for j in range(n-k) :
            delta[k+2*j] = (delta[k+2*j+1] - delta[k+2*j-1]) / (xi[k+j]-xi[j])
    return delta

def NewtonInterpol(xi,yi,a,b,nbEvalPts) :
    """ Polynomial interpolation of the data (xi,yi) in the Newton basis, 
        with Horner evaluation. 
        Returns a sampling of the interpolating polynomial over 
        the interval [a,b] with nbEvalPts points
        Input :
            xi, yi = two vectors of same size n+1 (n = degree)
            a,b = two real numbers with a < b
            nbEvalPts = integer = number of sampling points
        Output :
            py = vector of nbEvalPts reals 
                (the sampling of the interpolating polynomial over [a,b])        
    """
    degree = np.size(xi) - 1
    t = np.linspace(a,b,nbEvalPts)
    delta = diffdiv(xi,yi)
    py = delta[degree] * np.ones(nbEvalPts)
    for k in range(degree-1,-1,-1) :
        py = py * (t - xi[k]) + delta[k]
    return py
     
def PolygonAcquisition(color1,color2) :
    """ Mouse acquisition of a polygon
        right click to stop
    """
    x = []  # x is an empty list
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            plt.plot(xx,yy,color1,markersize=8)
            x.append(xx)
            y.append(yy)
            plt.draw()
            if len(x) > 1 :
                plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    return x,y

"""------------------------------------------------------
MAIN PROGRAM :
------------------------------------------------------"""
fig = plt.figure(1,(8,8))
ax = fig.add_subplot(111)
xlim = (-10, 10)
ylim = (-10, 10)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Acquisition of a polygon :
xi, yi = PolygonAcquisition('oc','c--')
degree = np.size(xi) - 1

# parameters domain :
a = 0
b = 1
nbt = 500

# 1) uniform parametrization ti :
ti = np.linspace(a, b, degree + 1) #car a = 0 et b = 1
# parametric interpolation according the uniform parametrization ti
tx = NewtonInterpol(ti,xi,a,b,1000)
ty = NewtonInterpol(ti,yi,a,b,1000)
plt.plot(tx, ty,'b',lw=1,label='uniform parameterization')


# 2) Chebyshev parametrization tch :
#
ti = [(a+b)/2 + (b-a)/2 * np.cos((2*i+1)*np.pi/(2*degree + 2)) for i in range(len(ti))]

# Chebyshev parametric interpolation :
tx = NewtonInterpol(ti,xi,a,b,1000)
ty = NewtonInterpol(ti,yi,a,b,1000)
#
plt.plot(tx, ty,'g',lw=1,label='Chebyshev parameterization')


# 3) chordal parameterization tc :
#
di = [np.sqrt((yi[i+1]-yi[i])**2+(xi[i+1]-xi[i])**2) for i in range(len(xi)-1)]
dt = np.sum(di)
ti = [a]
print(di)
for i in range(len(di)):
    ti.append(ti[i]+(di[i]/dt))

tx = NewtonInterpol(ti,xi,a,b,1000)
ty = NewtonInterpol(ti,yi,a,b,1000)

plt.plot(tx, ty,'r',lw=1,label='chordal parameterization')

plt.legend(loc='best') 

