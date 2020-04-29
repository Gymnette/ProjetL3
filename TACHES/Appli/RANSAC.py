# -*- coding: utf - 8 -*-
"""
Avril 2020

@author: Interpolaspline
"""

import sys

from random import sample
import numpy as np

import matplotlib.pyplot as plt
import plotingv2 as plot

import load_tests as ldt
import splines_naturelles as splnat
import splines_de_lissage as spllis


def repartition_equitable(x,n):
    """
    Repartit équitablement les données x en n intervalles

    Parameters
    ----------
    x : liste d'int
    n : int

    Returns
    -------
    rep : liste d'int

    """
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

def alea(n, nb):
    """
    Génère nb entiers aléatoires dans [0,n[

    Type entrées :
        n : int
        nb : int

    Type sorties :
        list[int] de longueur nb
    """
    return sample(list(range(n)), nb)

def trouver_ind(elem, liste):
    """
    Renvoie l'indice de l'élément de liste le plus proche de elem

    Type entrées :
        elem : float
        liste : list[float]

    Type sorties :
        int
    """
    for i, e in enumerate(liste):
        if e == elem:
            return i
        if e > elem:
            if i == 0:
                return i
            if e - elem < elem - liste[i - 1]:
                return i
            return i - 1
    return len(liste) - 1


def trouver_et_distance_ind_para(x, y, xc, yc, dist):
    """
    Renvoie la distance à la courbe de x,y, xc et yc représentant la courbe.

    Type entrées :
        x : float
        y : float
        xc : vecteur de float
        yc : vecteur de float
        dist : fonction : (float,float,float,float) -> float

    Type sorties :
        float
    """
    d = [dist(x, y, xc[i], yc[i]) for i in range(len(xc))]
    return min(d)

def calcul_erreur_courbe(xtest, ytest, xspline, yspline, dist):
    """
    Renvoie l'erreur totale des points à tester à la courbe, selon la fonction distance.

    Type entrées :
        xtest : vecteur de float
        ytest : vecteur de float
        xspline : vecteur de float
        yspline : vecteur de float
        dist : fonction : (float,float,float,float) -> float

    Type sorties :
        float
    """
    err = 0
    for i, e in enumerate(xtest):
        err += trouver_et_distance_ind_para(e, ytest[i], xspline, yspline, dist)
    return err

#########################
# Fonctions de distance #
#########################

def d_euclidienne(x0, y0, x1, y1):
    """
    Calcul de la distance euclidienne entre (x0,y0) et (x1,y1)

    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float

    Type sorties :
        float
    """
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

def d_carre(x0, y0, x1, y1):
    """
    Calcul de la distance euclidienne au carré entre (x0,y0) et (x1,y1)

    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float

    Type sorties :
        float
    """
    return d_euclidienne(x0, y0, x1, y1) ** 2

def calcul_Spline_NU(X, Y, a, b, n):
    '''
    Calcul de la spline interpolant les données dans l'intervalle [a,b],
    sur n points d'interpolation non uniformes
    Si a et / ou b ne sont pas présents dans X, la spline est prolongée
    de manière linéaire entre a et min(x), et entre b et max(x)
    La pente vaut alors sur ces intervalles respectivement la dérivée au point
    min(x) et la dérivée au point max(x).
    Renvoie xres et yres, des vecteurs de réels donnant
    la discrétisation de la spline.
    '''
    H = [X[i + 1] - X[i] for i in range(n - 1)]
    A = splnat.Matrix_NU(H)
    B = splnat.Matrix_NU_resulat(Y, H)
    Yp = np.linalg.solve(A, B)
    xres = []
    yres = []
    if X[0] != a:
        # y = cx + d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[0], Y[0])
        # On a alors y = YP[0] * x + (Y[0] - YP[0]X[0])
        xtemp = np.linspace(a, X[0], 100)
        ytemp = [Yp[0] * x + (Y[0] - Yp[0] * X[0]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp

    for i in range(0, n - 2):
        xtemp, ytemp = splnat.HermiteC1_non_affiche(X[i], Y[i], float(Yp[i]), X[i + 1], Y[i + 1], float(Yp[i + 1]))
        xtemp = list(xtemp)
        ytemp = list(ytemp)

        xres += xtemp
        yres += ytemp
    i = n - 2
    xtemp, ytemp = splnat.HermiteC1_non_affiche(X[i], Y[i], float(Yp[i]), X[i + 1], Y[i + 1], float(Yp[i + 1]))
    xtemp = list(xtemp)
    ytemp = list(ytemp)
    xres += xtemp
    yres += ytemp


    if X[-1] != b:
        # y = cx + d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[-1], Y[-1])
        # On a alors y = YP[-1] * x + (Y[-1] - YP[-1]X[-1])
        xtemp = np.linspace(X[-1], b, 100)
        ytemp = [Yp[-1] * x + (Y[-1] - Yp[-1] * X[-1]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp


    return xres, yres

def calcul_Spline_lissage(uk, zk, a, b, n, rho, mode=None):
    """
    Cette fonction calcule la spline de lissage des données uk zk sur l'intervalle [a,b].
    Elle considère n noeuds, répartis selon Chevyshev.
    Le paramètre de lissage est fourni à la fonction : rho.
    Renvoie la discrétisation de la spline de lissage.

    Type entrées :
        uk : vecteur de float
        zk : vecteur de float
        a : float
        b : float
        n : int
        rho : float

    Type sorties :
        list[float],list[float]

    """

    #Tri
    uk, zk = ldt.sortpoints(uk, zk)

    #if mode is None:
       # ldt.affiche_separation()
        #print("\nEntrez le mode de traitement du fichier :")
        #print("1 : Repartition uniforme des noeuds")
        #print("2 : Repartition de Chebichev")
        #print("3 : Repartition aléatoire")
        #print("4 : Repartition optimale")
        #mode = ldt.input_choice(['1', '2', '3','4'])

   # if mode == '1':
        #xi = np.linspace(a, b, n)
   # elif mode == '2':
       # xi = splnat.Repartition_chebyshev(a, b, n)
    #elif mode == '3':
        #Test sur une repartition des noeuds aleatoire
     #   xi = splnat.Repartition_aleatoire(a, b, n)
    #else:
     #   xi = spllis.Repartition_optimale(uk)
      #  n = len(xi)
    xi = repartition_equitable(uk,n)

    if spllis.presence_intervalle_vide(xi, uk):
        ldt.affiche_separation()
        print("\nErreur : Un intervalle vide est detecté.")
        print("Merci de changer au moins un des paramètres suivants :")
        print(" - nombre de noeuds")
        print(" - type de répartition\n")
        ldt.affiche_separation()
        sys.exit(1)

    H = [xi[i + 1] - xi[i] for i in range(len(xi) - 1)] # vecteur des pas de la spline

    Y = spllis.Vecteur_y(uk, [zk], xi, n, H, rho)
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(spllis.MatriceA(n, H), (np.dot(spllis.MatriceR(n, H), Y))))
    xx = []
    yy = []
    for i in range(n - 1):
        x, y = spllis.HermiteC1(xi[i], yi[0][i], yip[0][i], xi[i + 1], yi[0][i + 1], yip[0][i + 1])
        xx = np.append(xx, x)
        yy = np.append(yy, y)

    return xx, yy

##################################
# RANSAC: interpolation robuste #
##################################
# http://w3.mi.parisdescartes.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf.

def ransac_auto(x, y, err, dist, nbpoints, rho, pcorrect=0.99,mode=None):
    """
    Automatisation de l'algorithme de Ransac avec un modèle de splines cubiques :
    le calcul de la proportion de points aberrants (outlier) ou non (inlier)
    est mise à jour au fur et à mesure.
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données,
    de l'erreur maximum acceptée entre un point et la spline,
    de la fonction distance associée,
    du nombre de points pour créer la spline d'essai,
    du paramètre de lissage,
    et de la probabilité d'obtenir un résultat correct avec cet algorithme.
    xres et yres représentent la spline qui interpole au mieux les données
    d'après RANSAC, si exact est vraie, tous les calculs sont effectués
    à partir de la spline cubique d'interpolation.

    Type entrées :
        x : vecteur de float
        y : vecteur de float
        nbitermax : int
        err : float
        dist : function : (float,float)x(float,float) -> float
        nbcorrect : int
        nbpoints : int
    Type sorties :
        xres : vecteur de float
        yres : vecteur de float
    """
    para = False #préparation pour l'implémentation du paramétrique
    a = min(x)
    b = max(x)

    prop_inlier = 0
    nbitermax = 500 # Sera ajusté en fonction de prop_inlier
    nbcorrect = int(np.floor(prop_inlier * len(x)))

    # Sauvegarde du meilleur modèle trouvé et de l'erreur associée
    xmod = None
    ymod = None
    errmod = - 1

    k = 0
    while k <= nbitermax:
        # Choix d'un échantillon
        i_points = alea(len(x), nbpoints)
        i_points.sort()
        # i_points contient toujours le même nombre de points distcints,

        x_selec = []
        y_selec = []
        for e in i_points:
            x_selec.append(x[e])
            y_selec.append(y[e])

        # Création de la courbe à partir de l'échantillon
        if not para:
            x_selec, y_selec = ldt.sortpoints(x_selec, y_selec)
        xres, yres = [], []
        if para:
            xres, yres = calcul_Spline_para(x_selec, y_selec)
        else:
            xres, yres = calcul_Spline_NU(x_selec, y_selec, a, b, nbpoints)
        # Calcul des erreurs
        liste_inlier = list(i_points)
        for i, e in enumerate(x):
            if i in i_points:
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            i_associe = - 1
            d_courbe = - 1
            if para:
                d_courbe = trouver_et_distance_ind_para(x[i], y[i], xres, yres, dist)
            else:
                i_associe = trouver_ind(x[i], xres)
                d_courbe = dist(x[i], y[i], xres[i_associe], yres[i_associe])
            if d_courbe <= err:
                liste_inlier.append(i)

        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier !
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            liste_inlier.sort()
            x_pour_spline = []
            y_pour_spline = []
            for e in liste_inlier:
                x_pour_spline.append(x[e])
                y_pour_spline.append(y[e])
            if not para:
                x_pour_spline, y_pour_spline = ldt.sortpoints(x_pour_spline, y_pour_spline)
            xtemp, ytemp = 0, 0

            if para:
                xtemp, ytemp = calcul_Spline_para(x_pour_spline, y_pour_spline)
            else:
                xtemp, ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline, a, b, nbpoints, rho, 5)

            err_temp = - 1
            if para:
                err_temp = calcul_erreur_courbe(x_pour_spline, y_pour_spline, xtemp, ytemp, dist)
                #print(i_points, err_temp)
            else:
                err_temp = 0
                for i, e in enumerate(x_pour_spline):
                    i_associe = trouver_ind(x_pour_spline[i], xtemp)
                    err_temp += dist(x_pour_spline[i], y_pour_spline[i], xtemp[i_associe], ytemp[i_associe])

            if errmod == - 1 or err_temp < errmod:
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)

                plot.scatterdata(x,y,c="+r",new_fig=False,show=False,legend="Points aberrants")
                plot.scatterdata(x_pour_spline, y_pour_spline, c="+b", new_fig=False, show=False,legend="Points")

            # Que le modèle soit retenu ou non, on met à jour la proportion d'inliers et ce qui est associé

            if prop_inlier < len(liste_inlier) / len(x):
                # Il y a plus d'inlier que ce qu'on pensait
                prop_inlier = len(liste_inlier) / len(x)
                if prop_inlier == 1:
                    break
                    # On a trouvé un modèle correct, pour lequel
                    #il n'y aurait pas de points aberrants
                if np.log(1 - pcorrect) / np.log(1 - prop_inlier ** len(x)) < 0:
                    break
                nbitermax = int(np.floor(np.log(1 - pcorrect) / np.log(1 - prop_inlier ** len(x)))) # Sera ajusté en fonction de prop_inlier
                if nbitermax > 500:
                    nbitermax = 500
                nbcorrect = int(np.floor(prop_inlier * len(x)))
        k += 1

    plt.plot(xmod,ymod,label = "spline obtenue")
    return xmod, ymod

def Faire_Ransac(x, y, rho,err=0.5, f=None, para='1'):
    """
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    f : TYPE, optional
        DESCRIPTION. The default is None.
    para : TYPE, optional
        DESCRIPTION. The default is '1'.

    Returns
    -------
    xreel : TYPE
        DESCRIPTION.
    yreel : TYPE
        DESCRIPTION.

    """

    ldt.affiche_separation()
    print("La répartition équitable est utilisée, les autres choix ne sont pas disponibles pour RANSAC.")
    #print("\nEntrez le mode de traitement du fichier :")
    #print("1 : Repartition uniforme des noeuds")
    #print("2 : Repartition de Chebichev")
    #print("3 : Repartition aléatoire")
    #print("4 : Repartition optimale")
    #mode = ldt.input_choice(['1', '2', '3','4'])
    mode = '4'
    #if mode == '4':
     #   nconsidere = len(spllis.Repartition_optimale(x))
    #else:
    nconsidere = spllis.choisir_n()
    xres, yres = ransac_auto(x, y, err, d_euclidienne, nconsidere, rho)

    #plt.plot(xres, yres, "r")

    xreel, yreel = calcul_Spline_lissage(x, y, min(x), max(x), len(x), rho, mode)
    plot.plot1d1d(xres, yres, c="b", show=False, new_fig=False, legend="Spline obtenue")
    plt.title("Algorithme de Ransac")
    """
    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi, f(xi), new_fig=False, c='g',show=False,legend="signal")
    """
    plot.show()
    return xreel, yreel

def Lancer_Ransac():
    """
    Returns
    -------
    None.

    """

    ldt.affiche_separation()
    print("\nAlgorithme de RanSac")
    x, y, f, M, is_array, seed = ldt.charge_donnees()
    if is_array:
        ldt.affiche_separation()
        print("\nDéfinir un paramètre de lissage pour tous les fichiers ? (y = oui, n = non)")
        print("Si oui, et que vous choisissez ensuite le paramètre automatique,")
        print("alors ce paramètre sera recalculé pour chaque fichier.")
        rho_fixe = ldt.input_choice()

        if rho_fixe == 'y':
            ldt.affiche_separation()
            print("\nChoix automatique du paramètre de lissage ? (y = oui, n = non)")
            rho_auto = ldt.input_choice()
            if rho_auto == 'n':
                rho = spllis.choisir_rho([],[], 'n')

        Xtab = []
        Ytab = []
        for i in range(len(x)):
            if rho_fixe == 'n':
                rho = spllis.trouve_rho(x[i],y[i])
                print("\nLe paramètre de lissage automatique serait : ", rho)
                print("Choisir ce paramètre de lissage ? (y = oui, n = non)")
                rho_auto = ldt.input_choice()
                rho = spllis.choisir_rho(x[i],y[i], rho_auto)
            else:
                if rho_auto == 'y':
                    rho = spllis.choisir_rho(x[i],y[i])

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
            X, Y = Faire_Ransac(x, y, rho, seuil, f, M)
            Xtab.append(X)
            Ytab.append(Y)

    else:
        #M = ldt.charge_methodes()
        ldt.affiche_separation()
        rho = spllis.trouve_rho(x,y)
        print("\nLe paramètre de lissage automatique serait : ", rho)
        print("Choisir ce paramètre de lissage ? (y = oui, n = non)")
        rho_auto = ldt.input_choice()
        rho = spllis.choisir_rho(x,y, rho_auto)
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

        Xtab, Ytab = Faire_Ransac(x, y, rho, seuil, f, M)

    if seed is not None:
        ldt.affiche_separation()
        print("Graine pour la génération du signal : ", seed)
        ldt.affiche_separation()

if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")
