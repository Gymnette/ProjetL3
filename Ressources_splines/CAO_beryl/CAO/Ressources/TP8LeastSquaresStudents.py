# TP8LeastSquaresStudents.py
"""----------------------------------------------
Least squares approximation:
    by a polynomial of degree 'degree'
    with mouse acquisition
    and 
    calculation of the Pearson correlation coefficient
----------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

def MousePointsV3(color1) :
    """ V3 --> Right click stops the acquisition
        Acquisition of (x,y)-coordinates with the mouse
        such that the x-ccordinates are all different
        Output : 2 vectors x, y of same size
    """
    x = []  # x, y are an empty lists
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            # the x-coordinates must be all distinct:
            test = 1
            if np.size(x) > 0 :
                for j in range(np.size(x)) :
                    if xx == x[j] :
                        test = 0
            if test == 1 : # xx is different from all the other points
                plt.plot(xx,yy,color1,markersize=4)
                x.append(xx)
                y.append(yy)
                plt.draw()
    return x,y

def LeastSquares(xi,yi,degree):
    """ Determination of the best polynomial of degree "degree"
        approximating data (xi,yi) according the least squares method
        Returns :
            cf = the monomial coefficients of this polynomial
            res = the residual standard deviation
    """
    # TO BE COMPLETED
    # TO BE COMPLETED
    return cf, res

def CorrelCoef(xi,yi):
    """ Calculation of the Pearson correlation coefficient"
        associated with data (xi,yi)
        in case of 'degree = 1'
        Returns :
            corr = the Pearson correlation coefficient
    """
    # TO BE COMPLETED
    # TO BE COMPLETED
    return corr


"""------------------------------------------------------
MAIN PROGRAM :
------------------------------------------------------"""
if __name__ == '__main__':
    plt.cla()
    fig = plt.figure(1,(8,8))
    # (8,8) is a tuple for figsize = width and height in inches (optional)
    # default values are currently (8,6)
    # Access to default values in Matplotlib --> plt.rcParams
    ax = fig.add_subplot(111)
    minmax = 8
    xlim = (-minmax, minmax)
    ylim = (-minmax, minmax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    degree = 1
    # data mouse acquisition
    xi, yi =  MousePointsV3('oc')
    if np.size(xi) < degree + 1 :
        raise Exception ("Sorry, not enough data points acquired")

    #least squares approximation
    # TO BE COMPLETED
    # TO BE COMPLETED

    # Horner evaluation and plotting
    # TO BE COMPLETED
    # TO BE COMPLETED

    # Pearson correlation coefficient
    if degree == 1 :
        corr = CorrelCoef(xi,yi)
        print(corr)
