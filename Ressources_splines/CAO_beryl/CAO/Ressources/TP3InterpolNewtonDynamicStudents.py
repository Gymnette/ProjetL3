# TP3InterpolNewton.py
"""
TP3
"Dynamic" Lagrange interpolation in Newton basis with divided differences
Two parts :
    1) interpolation of a function f at n given uniform data 
    2) data points are dynamically acquired and interpolated
"""
import numpy as np
import matplotlib.pyplot as plt

"""--------------------
Test function
--------------------"""
def ftest(t) :
    y = (np.cos(1-t**2)) * np.exp(-t**2 + 3*t - 2)
    return y

def updateDD(xi,yi,delta,xnew,ynew) :
    """ Update the vector delta of divided differences (DD)
        when a new interpolation datum (xnew,ynew) 
        is added to the data xi,yi.
        The first half of the vector delta contains the DD 
        The second half contains the necessary data for subsequent updates
        --> Return :
            - vectors of updated data xi and yi (xnew, ynew at the end)
            - updated vector delta
    """
    n = np.size(xi)
    xi = np.append(xi,xnew)
    yi = np.append(yi,ynew)
    delta = np.append(delta,[0, ynew])
    for j in range(1,n+1) :
        k = 2*n-j
        delta[k] = (delta[k+1] - delta[k-1]) / (xi[n] - xi[n-j])
    return xi, yi, delta

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
    #Horner evaluation of the interpolation polynomial py
    py = delta[degree] * np.ones(nbEvalPts)
    for k in range(degree-1,-1,-1) :
        py = py * (t - xi[k]) + delta[k]
    return py

def AcquisitionOnePoint(f,a,b,color1) :
    """ Acquisition of one data interpolation point x in the interval [a,b]
        and plot of the 2D point (x, f(x))
        Output : 
            x, y=f(x) (a tuple of 2 real values)
    """
    coord = []
    while coord == [] :
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            x = coord[0][0]
            y = f(x)
            plt.plot(x,y,color1,markersize=6)
            plt.draw()
    return x, y

"""------------------------------------------------------
MAIN PROGRAM :
------------------------------------------------------"""    

"""------------------------------------------------------
TP3 QUESTION 1) :
    Interpolation in the Newton basis
    relative to uniformly distributed data
------------------------------------------------------"""
# Plot of the test function to be interpolated :
plt.cla()
a = 0
b = 4
nbt = 300
t = np.linspace(a,b,nbt)
y = ftest(t)
plt.plot(t, y, 'r--',lw=2)

# uniformly distributed interpolation points xi (i=0,1,...,n):
degree = 4
xi = np.linspace(a,b,degree+1)
yi = ftest(xi)
plt.plot(xi,yi,'or',ms=8)

# Newton interpolation
y = NewtonInterpol(xi,yi,a,b,nbt)
plt.plot(t, y,'b',lw=1)
plt.legend(['test function', 'equally spaced points',
            'interpolating polynomial'], loc='best')

    
"""------------------------------------------------------
TP3 QUESTION 2) :
    DYNAMIC INTERPOLATION
    data are dynamically acquired and interpolated
------------------------------------------------------"""
# Plot of the test function to be interpolated :
plt.cla()
a = 0
b = 4
nbt = 300
t = np.linspace(a,b,nbt)
y = ftest(t)
plt.plot(t, y, 'c--',lw=1,label='test function')
plt.axis([-0.05,4.05,-2.5,2.5]) # size of the axis

# DYNAMIC interpolation
# Initialization : interpolation at one point (degree = 0)
degree = 0
xi = np.array([])
yi = np.array([])
(x,y) = AcquisitionOnePoint(ftest,a,b,'ob')
xi = np.append(xi,x)
yi = np.append(yi,y)
delta =     #    TO BE COMPLETED...
NPoly = np.ones(np.size(t))
py =        #    TO BE COMPLETED...
plt.plot(t, py, lw=1, label="degree = "+str(degree))

degreeMax = 6
for degree in range(1, degreeMax+1) :
    # acquisition of a new datum
    #    TO BE COMPLETED...
    # update of the DD
    #    TO BE COMPLETED...
    # evaluation of the interpolation polynomial
    #    TO BE COMPLETED...
    #    TO BE COMPLETED...
    plt.plot(t, py, lw=1, label="degree = "+str(degree))   
plt.legend(loc='best')


"""------------------------------------------------------
TP3 QUESTION 3) :
    COMPARISON OF THESE TWO APPROACHES
------------------------------------------------------"""
"""

TO BE COMPLETED...


"""

