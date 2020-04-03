
import numpy as np

from scipy import linalg



def construct(x, y, rho = .005):
    """
    Création du vecteur yest depuis y, où ses valeurs sont estimées par le poid respectif de chacune
    
    Intput :
        uk,zk : vecteurs de float de l'échantillon étudié
     
    Output :
        yest :  vecteurs de float(valeurs en y) de l'échantillon étudié, estimés par la méthode LOESS. 
    """
    
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([np.exp(- (x - x[i])**2/(2*rho)) for i in range(n)])     
   
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest