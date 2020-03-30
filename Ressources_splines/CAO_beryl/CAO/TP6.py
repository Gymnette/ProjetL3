# TP6HermiteInterpolationStudents.py
"""
Cubic and Quintic Hermite interpolation.
Application in filling of holes
Comparizon between 
    - cubic Hermite C1 interpolation 
    - quintic Hermite C2 interpolation 
"""
import numpy as np
import matplotlib.pyplot as plt

# Cubic Hermite basis over [0,1] :
def H0(t) :
    return 1 - 3 * t**2 + 2 * t**3
def H1(t) :
    return t - 2 * t**2 + t**3  
def H2(t) :
    return - t**2 + t**3
def H3(t) :
    return 3 * t**2 - 2 * t**3

# Quintic Hermite interpolation basis over [0,1]
def  Q0(t) :
    return - 6 * t**5 + 15 * t**4 - 10 * t**3 + 1
def  Q1(t) :
    return - 3 * t**5 + 8 * t**4 - 6 * t**3 + t
def  Q2(t) :
    return (- t**5 + 3 * t**4 - 3 * t**3 + t**2) / 2
def  Q3(t) :
    return ( t**5 - 2 * t**4 + t**3) / 2
def  Q4(t) :
    return - 3 * t**5 + 7 * t**4 - 4 * t**3
def  Q5(t) :
    return 6 * t**5 - 15 * t**4 + 10 * t**3

# Cubic Hermite C1 interpolation over 2 points
def HermiteC1(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    t = (x-x0)/(x1-x0)
    y = y0 * H0(t) + y0p * (x1-x0) * H1(t) + y1p * (x1-x0) * H2(t) + y1 * H3(t)
    plt.plot(x,y,'c--',lw=2,label='cubic interpolant')

# Quintic Hermite interpolation over 2 points
def HermiteC2(x0,y0,y0p,y0pp,x1,y1,y1p,y1pp):
    """ Quintic Hermite interpolation of order two 
        (interpolation of value + first and second derivatives)
        over 2 points x0 < x1
        Input :
            x0,y0,y0p,y0ppx1,y1,y1p,y1pp = Hermite data of order 2
        Return :
            plot the quintic Hermite interpolant
    """
    x = np.linspace(x0,x1,200)
    t = (x-x0)/(x1-x0)
    y = y0 * Q0(t) + y0p * (x1-x0) * Q1(t) + y0pp * ((x1-x0)**2) * Q2(t) + y1pp * ((x1-x0)**2) * Q3(t) + y1p * (x1-x0) * Q4(t) + y1 * Q5(t)
    plt.plot(x,y,'y',lw=2,label='quintic interpolant')

    
# Test functions and their first and second derivatives
def f1(x):
    return (np.exp(x) / 2) - 1
def f1p(x):
    return np.exp(x) / 2
def f1pp(x):
    return np.exp(x) / 2

def f2(x):
    return np.sin(x**2)-1
def f2p(x):
    return 2*x*np.cos(x**2)
def f2pp(x):
    return (2*np.cos(x**2)-4*np.sin(x**2)*x**2)

def f3(x):
    return -1 + 2*(np.sin(2*x))/x
def f3p(x):
    return (4*x*np.cos(2*x)-2*np.sin(2*x))/(x**2)
def f3pp(x):
    return -8*np.sin(2*x)/x - 8*np.cos(2*x)/(x**2) + 4*np.sin(2*x)/(x**3)

"""------------------------------------------------------
MAIN PROGRAM :
------------------------------------------------------"""
#plt.clf()

# The three curves on three distinct intervals
a = -1 ; b = 0.5
t1 = np.linspace(a,b,150)
y1 = f1(t1) 
plt.plot(t1,y1,lw=3)

c = 2; d = 3
t2 = np.linspace(c,d,100)
y2 = f2(t2)
plt.plot(t2,y2,lw=3)

e = 5; f = 8
t3 = np.linspace(e,f,300)
y3 = f3(t3)
plt.plot(t3,y3,lw=3)


# Filling holes :
#   cubic Hermite interpolation ==> C1 contact 
#   quintic Hermite interpolation ==> C2 contact 

HermiteC1(1/2,f1(1/2),f1p(1/2),2,f2(2),f2p(2))
HermiteC1(3,f2(3),f2p(3),5,f3(5),f3p(5))
HermiteC2(1/2,f1(1/2),f1p(1/2),f1pp(1/2),2,f2(2),f2p(2),f2pp(2))
HermiteC2(3,f2(3),f2p(3),f2pp(3),5,f3(5),f3p(5),f3pp(5))

plt.legend(loc='best')
