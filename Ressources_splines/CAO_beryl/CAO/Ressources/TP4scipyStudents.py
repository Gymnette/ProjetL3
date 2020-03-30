# TP4scipyStudents.py
"""----------------------------------------
Interpolation with scipy (scientific Python)
  --> function scipy.interpolate.lagrange()
      in the monomial basis (may becomes unstable if the degree increases)
  --> comparizon with Newton interpolation
-------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def f1(t):
    return np.cos(t**2)

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

"""------------------------------------------------------
MAIN PROGRAM :
    Scipy interpolation (in the monomial basis)
    and comparizon with Newton interpolation
------------------------------------------------------"""
plt.clf()
a = 0
b = 3
degree = 6
# the function to be interpolated :
t = np.linspace(a, b, 200)
plt.plot(t, f1(t), label="f(t) = cos(t^2)")
# the data points :
xi = np.linspace(a, b, degree+1)
yi = f1(xi)
plt.plot(xi,yi,'ro')

"""----------
Interpolation with scipy :
----------"""
cf = interpolate.lagrange(xi, yi)
# coefficients of the interpolating polynomial in monomial basis : 
# cf[k] is the coefficient of x^k

# Horner evaluation :
#
#    TO BE COMPLETED .....
#
plt.plot(t, p, label="Uniform scipy interpolant")

"""----------
Comparizon with Newton interpolation
----------"""
# Uniform interpolation with Newton and DD
#
#    TO BE COMPLETED .....
#
plt.plot(t, p, label='uniform Newton interpolant')

plt.legend(loc="best")
