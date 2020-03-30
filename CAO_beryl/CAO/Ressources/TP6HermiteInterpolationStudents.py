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
    #
    # TO BE COMPLETED 
    #
    plt.plot(x,y,'c--',lw=2,label='cubic interpolant')

# Quintic Hermite interpolation over 2 points
def HermiteC2a(x0,y0,y0p,y0pp,x1,y1,y1p,y1pp):
    """ Quintic Hermite interpolation of order two 
        (interpolation of value + first and second derivatives)
        over 2 points x0 < x1
        Input :
            x0,y0,y0p,y0ppx1,y1,y1p,y1pp = Hermite data of order 2
        Return :
            plot the quintic Hermite interpolant
    """
    x = np.linspace(x0,x1,200)
    #
    # TO BE COMPLETED 
    #
    #
    plt.plot(x,y,'y',lw=2,label='quintic interpolant')

    
# Test functions and their first and second derivatives
def f1(x):
    return (np.exp(x) / 2) - 1
def f1p(x):
    return np.exp(x) / 2
def f1pp(x):
    return np.exp(x) / 2

#
# TO BE COMPLETED 
#
#

#
# TO BE COMPLETED 
#
#

"""------------------------------------------------------
MAIN PROGRAM :
------------------------------------------------------"""
plt.clf()

# The three curves on three distinct intervals
a = -1 ; b = 0.5
t = np.linspace(a,b,100)
y = f1(t) 
plt.plot(t,y,lw=3)

#
# TO BE COMPLETED 
#
#

#
# TO BE COMPLETED 
#
#

# Filling holes :
#   cubic Hermite interpolation ==> C1 contact 
#   quintic Hermite interpolation ==> C2 contact 

#
# TO BE COMPLETED 
#
#

#
# TO BE COMPLETED 
#
#

plt.legend(loc='best')
