# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:58:21 2020

@author: Interpolaspline
"""
import sys

from typing import List
import numpy as np
import math
import matplotlib.pyplot as plt

import splines_de_lissage as spllis
import load_tests as ldt
import plotingv2 as plot


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
    :return: retourne l'indice du point aberrant
    """

    ind = -1
    dmax = -math.inf
    for i in range(len(y)):
        d = voisinsI(X, Y, x[i], y[i])
        if d > seuil and d > dmax:
                ind = i
                dmax = voisinsI(X, Y, x[i], y[i])

    return ind

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

        Theta = np.linalg.solve(A, b)
        y_estimated[i] = Theta[0] + Theta[1] * x[i]

    return y_estimated

def repartition_equitable(x,n):
    ninter = n-1
    nb = len(x)//ninter
    plus_un = len(x)%ninter

    rep = []
    ind = 0
    for _ in range(n-1): # Pour chaque partie
        rep.append(x[ind])
        ind += 1
        compteur = 1
        while compteur != nb :
            compteur += 1
            ind += 1
        if plus_un != 0 :
            plus_un -= 1
            ind += 1
    rep.append(x[-1])
    return rep

def Faire_intuitive(uk, zk, f, mode):

    a = min(uk)
    b = max(uk)
    if mode == 4:
        n = spllis.choisir_n()

    ldt.affiche_separation()
    print("Choisissez le seuil d'erreur (réel positif) :")

    seuil = -1
    while seuil <=0 :
        seuil = input("> ")
        if seuil == "q":
            sys.exit(0)
        else:
            try :
                seuil = float(seuil)
            except ValueError:
                print("Choisissez un nombre valide")
                seuil = -1

    plt.figure()
    plt.plot(uk,zk,"+")
    uk = list(uk)
    zk = list(zk)

    stop = 0
    while stop<1000: #On break cette boucle lorsqu'il n'y a plus de points aberrants

        # A enlever
        #plt.figure()
        #plt.plot(uk,zk,"+")


        #test précis :
        if mode == '1':
            xi = np.linspace(a, b, n)
        elif mode == '2':
            xi = spllis.Repartition_chebyshev(a, b, n)
        elif mode == '3':
            #Test sur une repartition des noeuds aleatoire
            xi = spllis.Repartition_aleatoire(a, b, n)
        else:
            xi = spllis.Repartition_optimale(uk)
            n = len(xi)
        # La répartition peut aussi être faite de manière aléatoire.
        # ATTENTION AUX INTERVALLES VIDES AVEC D'AUTRES REPARTITIONS (toujours le même problème)
        #xi = Repartition_chebyshev(a,b,n)
        #plt.scatter(xi,[0]*n,label = 'noeuds')

        H = [xi[i+1]-xi[i] for i in range(len(xi)-1)] # vecteur des pas de la spline
        #rho = spllis.trouve_rho(uk, zk)
        rho = 0.04
        #print(H03(N,n,uk,xi,H))

        Y = spllis.Vecteur_y(uk,[zk],xi,n,H,rho)
        yi = np.transpose(Y)
        yip = np.transpose(np.linalg.solve(spllis.MatriceA(n,H),(np.dot(spllis.MatriceR(n,H),Y))))
        xx=[]
        yy=[]
        for i in range(n-1):
            x,y = spllis.HermiteC1(xi[i],yi[0][i],yip[0][i],xi[i+1],yi[0][i+1],yip[0][i+1])
            xx=np.append(xx,x)
            yy=np.append(yy,y)

        ind_le_plus_aberrant = Erreur(uk,zk,xx,yy,seuil)

        #y_estim = erreur.Erreur_poids(uk,zk,xx,yy,seuil,1/100000,rho)
        #plt.plot(xx,yy,lw=1,label='spline de lissage avec rho = '+str(rho))
        if ind_le_plus_aberrant == -1 or len(uk) <= n+1 :
            plt.plot(xx,yy)
            break

        plt.plot(uk[ind_le_plus_aberrant], zk[ind_le_plus_aberrant], 'or',color='y', label='donnée retirée, seuil = '+str(seuil))
        #plt.plot(xx,yy)


        #plt.plot(uk,y_estim,color='r',label='avec poids')

        uk.pop(ind_le_plus_aberrant)
        zk.pop(ind_le_plus_aberrant)
        stop+=1

    ldt.affiche_separation()
    print("Points aberrants trouvés : ",stop)
    ldt.affiche_separation()

    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi, f(xi), new_fig=False, c='g')
    plt.show()


def Lancer_intuitive():
    """
    Lance la methode intuitive
    """

    D_meth = {'1': "repartition uniforme des noeuds",
              '2': "repartition de Chebichev",
              '3': "repartition aléatoire",
              '4': "repartition optimale"}

    uk, zk, f, M, is_array, seed = ldt.charge_donnees(D_meth)
    if seed is not None:
        print("Graine pour la génération du signal : ", seed)

    if is_array:
        for i, xi in enumerate(uk):
            Faire_intuitive(xi, zk[i], f, M)
    else:
        Faire_intuitive(uk, zk, f, M)
