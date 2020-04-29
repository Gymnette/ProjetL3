# -*- coding: utf - 8 -*-
"""
Avril 2020

@author: Interpolaspline
"""

import numpy as np
import load_tests as ldt
import plotingv2 as plot
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import splines_de_lissage as spllis





def Calculer_LOESS_robuste(x, y,rho,iter=10):

    """
    Création du vecteur y_estimated depuis y, où ses valeurs sont estimées par le poids de chaque point donné par nos méthodes de détection

    Intput :
        uk,zk : vecteurs de float de l'échantillon étudié
        rho : paramètre de lissage
        iter : iterations de robustesse
    Output :
        y_estimated :  vecteurs de float(valeurs en y) de l'échantillon étudié, estimés par la méthode LOESS.
    """

    n = len(x)
    w = np.array([np.exp(- (x - x[i])**2/(2*rho)) for i in range(n)])
    y_estimated = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            Theta = linalg.solve(A, b)
            y_estimated[i] = Theta[0] + Theta[1] * x[i]

        erreurs = y - y_estimated
        s = np.median(np.abs(erreurs))
        delta = np.clip(erreurs / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return y_estimated



def Faire_LOESS_robuste(x, y, rho):
    """
    Parameters
    ----------
    x : liste d'int
    y : liste d'int
    rho : float
    """

    #Tri
    x, y = ldt.sortpoints(x, y)

    a = min(x) # intervalle
    b = max(y) # intervalle


    ldt.affiche_separation()
    print("\nEntrez le mode de traitement du fichier :")
    print("1 : Repartition uniforme des noeuds")
    print("2 : Repartition de Chebichev")
    print("3 : Repartition aléatoire")
    print("4 : Repartition optimale")
    mode = ldt.input_choice(['1', '2', '3','4'])

    if mode == '4':
        n = len(spllis.Repartition_optimale(x))
    else:
        n = spllis.choisir_n()

    print("\nAfficher les noeuds ? (y = oui, n = non)")
    aff_n = ldt.input_choice()

    yres = Calculer_LOESS_robuste(x, y,rho)

    plot.scatterdata(x, y, c= "rx", show=False, new_fig=False, legend="Echantillon")



    if mode == '1':
        xi = np.linspace(a, b, n)
    elif mode == '2':
        xi = spllis.Repartition_chebyshev(a, b, n)
    elif mode == '3':
        #Test sur une repartition des noeuds aleatoire
        xi = spllis.Repartition_aleatoire(a, b, n)
    else:
        xi = spllis.Repartition_optimale(x)
        n = len(xi)


    if spllis.presence_intervalle_vide(xi, x):
        ldt.affiche_separation()
        print("\nErreur : Un intervalle vide est detecté.")
        print("Merci de changer au moins un des paramètres suivants :")
        print(" - nombre de noeuds")
        print(" - type de répartition\n")
        ldt.affiche_separation()
        sys.exit(1)



    if aff_n == 'y':
        plt.scatter(xi, [0] * len(xi), label='noeuds')

    H = [xi[i + 1] - xi[i] for i in range(len(xi) - 1)] # vecteur des pas de la spline

    Y = spllis.Vecteur_y(x, [yres], xi, n, H, rho)
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(spllis.MatriceA(n, H), (np.dot(spllis.MatriceR(n, H), Y))))
    xx = []
    yy = []
    for i in range(n - 1):
        x, y = spllis.HermiteC1(xi[i], yi[0][i], yip[0][i], xi[i + 1], yi[0][i + 1], yip[0][i + 1])
        xx = np.append(xx, x)
        yy = np.append(yy, y)
    plt.plot(xx, yy, lw=1, label='spline de lissage avec rho = ' + str(round(rho, 3)))


    plt.title("LOESS Robuste")
    plt.legend()
    plt.show()


def Lancer_LOESS_robuste():
    """
    Returns
    -------
    None.

    """
    x, y, f, M, is_array, seed = ldt.charge_donnees()
    print("\nAlgorithme de LOESS Robuste")

    rho = spllis.choisir_rho(x,y)
    ldt.affiche_separation()
    print("\nLe paramètre de lissage automatique serait : ", rho)
    print("Choisir ce paramètre de lissage ? (y = oui, n = non)")
    rho_auto = ldt.input_choice()
    rho = spllis.choisir_rho(x,y, rho_auto)

    Faire_LOESS_robuste(x, y, rho)


if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer appli_interpolaspline.")
