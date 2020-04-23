
import numpy as np
from scipy import linalg

def poids_faibles(x, y,v_poids,rho):
    
    """
    Création du vecteur y_estimated depuis y, où ses valeurs sont estimées par le poid respectif de chacune
    
    Intput :
        uk,zk : vecteurs de float de l'échantillon étudié
        v_poids : vecteur de float, poids des valeurs de l'échantillon
    Output :
        y_estimated :  vecteurs de float(valeurs en y) de l'échantillon étudié, estimés par la méthode LOESS. 
    """

    n = len(x)
    y_estimated = np.zeros(n)
   

    w = np.array([np.exp(- (x - x[i])**2/(2*rho))*v_poids[i] for i in range(n)])  #initialise tous les poids    

    for i in range(n): #Calcule la nouvelle coordonnée de tout point
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        Theta = linalg.solve(A, b)
        y_estimated[i] = Theta[0] + Theta[1] * x[i] 
            
    
    return y_estimated







