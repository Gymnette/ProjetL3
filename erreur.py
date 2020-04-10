from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



def voisinsI(x, y, a, b):
    """
    :param x: une liste de réels d'abscisses
    :param y: une liste de réels d'ordonnées
    :param a: un réel abscisse
    :param b: un réel ordonné
    :return: la distance de point (a,b) au plus proche point dans x,y
    """
    n = len(y)
    l = []
    for j in range(n):
        if y[j] != b and a != x[j]:
            l.append(np.sqrt((b - y[j])**2 + (a - x[j])**2))
    l.sort()
    return l[0]

def mini(x, y, X, Y):
    l = []
    for i in range(len(y)):
        l.append(voisinsI(X, Y, x[i], y[i]))
    l.sort()
    return l


def Erreur(x, y, X, Y, seuil):

    """
    :param x: une liste de réels d'abscisses
    :param y: une liste de réels d'ordonnées
    :param X: une liste de réels d'abscisses de la fonction
    :param Y: une liste de réels d'ordonnées de la fonctio
    :param seuil: un réel à fixer selon l'echantillon
    :return: retourne 4 listes : 2 pour les abscisses et ordonnées
             contenant les valeurs aberrantes et les 2 autres celles
             non aberrantes
    """

    lax, lay, lnax, lnay = [],[],[],[]

    for i in range(len(y)):
        d = voisinsI(X, Y, x[i], y[i])
        if d > seuil:
            lay.append(y[i])
            lax.append(x[i])
        else:
            lnay.append(y[i])
            lnax.append(x[i])
    return lax, lay,lnax, lnay

def Erreur_poids(x, y, X, Y, seuil,poids,rho):
    """
    :param x: une liste de réels d'abscisses
    :param y: une liste de réels d'ordonnées
    :param X: une liste de réels d'abscisses de la fonction
    :param Y: une liste de réels d'ordonnées de la fonctio
    :param seuil: un réel à fixer selon l'echantillon
    :param poids : un réel : le poids des points aberrants
    :return: retourne 2 listes : une des indices des points aberrants
    et l'autre la liste des poids
    """

    l_poids: List[int] = [1]*len(y)

    for i in range(len(y)):
        d = voisinsI(X, Y, x[i], y[i])
        if d > seuil:
            l_poids.append(poids)
    return construct(x, y, l_poids, rho)


def construct(x, y, v_poids, rho=0.05):
    """
    Création du vecteur y_estimated depuis y, où ses valeurs sont estimées par le poid respectif de chacune

    Intput :
        uk,zk : vecteurs de float de l'échantillon étudié

    Output :
        y_estimated :  vecteurs de float(valeurs en y) de l'échantillon étudié, estimés par la méthode LOESS.
    """
    if rho == 0:
        rho = 0.001
    n = len(x)
    y_estimated = np.zeros(n)

    w = np.array(
        [np.exp(- (x - x[i]) ** 2 / (2 * rho)) * v_poids[i] for i in range(n)])  # initialise tous les poids

    for i in range(n):  # Calcule la nouvelle coordonnée de tout point
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        Theta = linalg.solve(A, b)
        y_estimated[i] = Theta[0] + Theta[1] * x[i]

    return y_estimated


if __name__ == '__main__':
    #y = [5, 7, 10, 15, 19, 21, 21, 21, 21, 23, 23, 23, 24, 25]
    y = [-13,16,23,7,9,11,10,13,5,3,30]
    x = [i for i in range(len(y))]
    X = [j for j in range(-10,14)]
    Y = [X[j]**2 for j in range(len(X))]
    """x,y = np.loadtxt('data.txt')
    print(np.abs(np.std(y)/np.std(x)))"""
    seuil = np.mean(mini(x,y,X,y))/np.std(mini(x,y,X,y))
    lax,lay,lnax,lnay = Erreur(x,y,X,Y,seuil)
    a = construct(X,Y,0.1)
    plt.figure('test erreur')
    plt.plot(lnax,lnay,'rx',color='r',label="donnees non aberrantes")
    plt.plot(lax, lay, 'or', color='b', label="donnees non aberrantes")
    plt.plot(X,Y, color='g', label="parabole")
    plt.legend(loc=True)
    plt.show()
