# !/ usr / bin / env python3
# -*- coding: utf - 8 -*-
"""
Created on Mon Mar 30 11:01:41 2020

@author: Amelys Rodet
"""

from random import random
import numpy as np
import matplotlib.pyplot as plt

import load_tests as ldt

import plotingv2 as plot


def H0(t):
    return 1 - 3 * t ** 2 + 2 * t ** 3
def H1(t):
    return t - 2 * t ** 2 + t ** 3
def H2(t):
    return - t ** 2 + t ** 3
def H3(t):
    return 3 * t ** 2 - 2 * t ** 3

# Cubic Hermite C1 interpolation over 2 points
def HermiteC1(x0, y0, y0p, x1, y1, y1p, label="", color='g'):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0, x1, 100)
    y = []
    for t in x:
        som = 0
        som += y0 * H0((t - x0) / (x1 - x0))
        som += y0p * (x1 - x0) * H1((t - x0) /(x1 - x0))
        som += y1p * (x1 - x0) * H2((t - x0) /(x1 - x0))
        som += y1 * H3((t - x0) / (x1 - x0))
        y.append(som)
    if label != "":
        plt.plot(x, y, color, lw=2, label=label)
    else:
        plt.plot(x, y, color, lw=2)

def HermiteC1_non_affiche(x0, y0, y0p, x1, y1, y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            x,y : cubic Hermite interpolant
    """
    x = np.linspace(x0, x1, 100)
    y = []
    for t in x:
        som = 0
        som += y0 * H0((t - x0) / (x1 - x0))
        som += y0p * (x1 - x0) * H1((t - x0) /(x1 - x0))
        som += y1p * (x1 - x0) * H2((t - x0) /(x1 - x0))
        som += y1 * H3((t - x0) / (x1 - x0))
        y.append(som)
    return(x, y)


def Equirepartis(a, b, n):
    '''
    Retourne un tableau de points équidistants de l'intervalle [a,b]
    '''
    T = np.linspace(a, b, n)
    return T

def Non_uniforme(a, b, n):
    '''
    Retourne un tableau de points répartis aléatoirement de l'intervalle [a,b],
    avec T[0] = a, T[n - 1] = b
    '''
    T = []
    T.append(a)
    for _ in range(n - 2):
        T.append(random() * (b - a) + a)
    T.sort()
    T.append(b)
    return T

def Matrix_NU(H):
    '''
    Renvoie la matrice associée au calcul des dérivées non uniformes
    '''
    n = len(H)
    Dm1 = [H[i] for i in range(1, n)]
    Dm1.append(1)
    Dm2 = np.array(Dm1)

    D1 = [2 * (H[i - 1] + H[i]) for i in range(1, n)]
    D1.append(2)
    D1.insert(0, 2)
    D2 = np.array(D1)

    Dp1 = [H[i] for i in range(0, n - 1)]
    Dp1.insert(0, 1)
    Dp2 = np.array(Dp1)

    A = np.diag(D2)
    B = np.diag(Dm2, - 1)
    C = np.diag(Dp2, 1)

    return A + B + C

def Matrix_NU_resulat(Y, H):
    '''
    Renvoie le second membre du système de calcul des dérivées uniformes
    '''
    n = len(H)
    R = [3 * (H[i - 1] / H[i] * (Y[i + 1] - Y[i]) + H[i] / H[i - 1] * (Y[i] - Y[i - 1])) for i in range(1, n)]
    R.insert(0, 3 / H[0] * (Y[1] - Y[0]))
    R.append(3 / H[n - 1] * (Y[n] - Y[n - 1]))
    R = np.mat(R)
    return R.transpose()

def Affiche_Spline_NU(X, Y, n, label="", color='r'):
    '''
    Affichage de la spline interpolant la fonction f dans l'intervalle [a,b],
    sur n points d'interpolation non uniformes

    Effet de bord: Affichage de la spline

    Aucun retour
    '''
    H = [X[i + 1] - X[i] for i in range(n - 1)]
    plt.scatter(X, Y, s=75, c='red', marker='o', label="NU interpolation points")
    A = Matrix_NU(H)
    B = Matrix_NU_resulat(Y, H)
    Yp = np.linalg.solve(A, B)
    for i in range(0, n - 2):
        HermiteC1(X[i], Y[i], float(Yp[i]), X[i + 1], Y[i + 1], float(Yp[i + 1]), label='', color=color)
    i = n - 2
    HermiteC1(X[i], Y[i], float(Yp[i]), X[i + 1], Y[i + 1], float(Yp[i + 1]), label, color)


def Affiche_Spline_Para(a, b, X, Y, label="", color="r", type_repartition=""):
    '''
    Affichage de la courbe paramétrique spline interpolant les points dans l'intervalle [a,b],
    sur n points d'interpolation répartis uniformément, façon chebyshev ou chordale

    Effet de bord: Affichage de la spline

    Aucun retour
    '''
    n = len(X)
    plt.scatter(X, Y)
    if type_repartition == "chordale":
        T = Repartition_cordale(X, Y, a, b)
    elif type_repartition == "chebyshev":
        T = Repartition_chebyshev(a, b, n)
    else:
        T = Equirepartis(0, 1, n)
    #Spline des (ti, xi)
    H = [T[i + 1] - T[i] for i in range(n - 1)]
    A = Matrix_NU(H)
    B = Matrix_NU_resulat(X, H)
    Xp = np.linalg.solve(A, B)
    Sx = []
    for i in range(0, n - 1):
        Sx.append(HermiteC1_non_affiche(T[i], X[i], float(Xp[i]), T[i + 1], X[i + 1], float(Xp[i + 1])))
    #Spline des (ti, yi)
    A = Matrix_NU(H)
    B = Matrix_NU_resulat(Y, H)
    Yp = np.linalg.solve(A, B)
    Sy = []
    for i in range(0, n - 1):
        Sy.append(HermiteC1_non_affiche(T[i], Y[i], float(Yp[i]), T[i + 1], Y[i + 1], float(Yp[i + 1])))
    #affichage
    for i in range(0, n - 2):
        plt.plot(Sx[i][1], Sy[i][1], color)
    plt.plot(Sx[n - 2][1], Sy[n - 2][1], color, label=label)

def Repartition_cordale(X, Y, a, b):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis façon chordale
    '''
    n = len(X)
    D = [np.sqrt((X[i + 1] - X[i]) ** 2 + (Y[i + 1] - Y[i]) ** 2) for i in range(n -1)]
    T = [a]
    som = np.sum(D)
    for i in range(n - 1):
        T.append(T[i] + D[i] / som)
    T.append(b)
    return T

def Repartition_chebyshev(a, b, n):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis façon chebyshev
    '''
    T = []
    t1 = float((a + b)) /2
    t2 = float((b - a)) /2
    for i in range(n):
        T.append(t1 + t2 * (np.cos((2 * i + 1) * np.pi /(2 * n + 2))))
    return T

def Repartition_aleatoire(a, b, n):
    rdm = np.random.rand(n - 2)
    rdm.sort()
    xi = [a]
    xi = np.append(xi, rdm * (b - a) + a)
    xi = np.append(xi, [b])
    for i in range(len(xi) - 1):
        if xi[i] == xi[i + 1]:
            if i == len(xi) - 2:
                xi[i] = xi[i + 1] + xi[i -1] /2
            else:
                xi[i + 1] = (xi[i] + xi[i + 2]) /2
    return xi

def test_fichier(U, Z, f=None, mode=None):

    if mode is None:
        #Demande du mode de traitement du fichier
        ldt.affiche_separation()
        print("\nEntrez le mode de traitement du fichier :", "1. tel quel", "2. tri sur l'axe X", "3. tri sur l'axe Y", sep='\n')
        mode = ldt.input_choice(['1', '2', '3'])

    #Affectations des extremums et du nombre de point
    #Independant du mode choisi
    a = min(U)
    b = max(U)
    n = len(U)

    if mode == '1':
        #Mode "tel quel" : les donnees sont prises dans l'ordre, en courbe paramétrique
        plt.figure()

        Affiche_Spline_Para(a, b, U, Z, label="pas de tri", color="r")

        plt.legend(fontsize="10")

    elif mode == '2':
        #Mode tri sur l'axe X : les donnees sont triees selon l'axe X
        #plt.figure()
        #Tri sur X
        X, Y = ldt.sortpoints(U, Z)

        Affiche_Spline_NU(X, Y, n, label="tri selon l'axe des abcisses", color='b')

        plt.legend(fontsize="10")

    elif mode == '3':
        #Mode tri sur l'axe X : les donnees sont triees selon l'axe Y
        #plt.figure()
        #Tri sur Y
        Y, X = ldt.sortpoints(Z, U)

        Affiche_Spline_Para(a, b, X, Y, label="tri selon l'axe des ordonnées", color="g")

        plt.legend(fontsize="10")

    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi, f(xi), new_fig=False, c='g')

    plot.show()

    ldt.affiche_separation()
    print("Spline créée !")
    ldt.affiche_separation()

def creation_spline_naturelle(x=None, y=None, f=None, is_array=False):

    print("\nCreation de la spline naturelle interpolant chaque point donne.\n")

    D_meth = {'1': "tel quel",
              '2': "tri sur l'axe X",
              '3': "tri sur l'axe Y"}
    M = None

    if (x is None) or (y is None):
        x, y, f, M, is_array, seed = ldt.charge_donnees(D_meth)
    elif is_array:
        M = ldt.charge_methodes(D_meth)

    if is_array:
        for i in range(len(x)):
            test_fichier(x[i], y[i], f, M)
    else:
        test_fichier(x, y, f, M)

    if seed is not None:
        print("Graine pour la génération du signal : ", seed)
        ldt.affiche_separation()
    print("Retour au menu principal...")
    ldt.affiche_separation()

if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")

