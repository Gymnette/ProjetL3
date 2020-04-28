# -*- coding: utf-8 -*-
import math
import gestion_aberrance as det
import load_tests as ldt
import Tache_4_methodes as meth
import plotingv2 as plot
import numpy as np


def winsorization(x, y, quantite_aberrante):
    """
    Applique la Winsorization de manière rigoureuse, selon la définition.
    quantite_aberrante appartient à [0,1[.

    Type entrées :
        y : vecteur de float
        quantite_aberrante : float

    Type sorties :
        liste modifiee
    """

    ybis = list(y)
    xabmin = []
    xabmax = []
    minival = []
    maxival= []

    n = len(y)
    nb = math.floor(quantite_aberrante*n)//2
    # On divise par deux car on réparti la quantité à retirer en haut et en bas.
    # nb vaut 0 si n*quantite_aberrante < 1.
    if nb == 0 :
        return

    # Listes de couples : (valeur,indice)
    donnees = [(y[i],i) for i in range(len(y))]
    donnees.sort()
    mini = [donnees.pop(0) for _ in range(nb)]
    maxi = [donnees.pop(-1) for _ in range(nb)]
    minimum = donnees[0][0]
    maximum = donnees[-1][0]

    # On affecte le minimum non aberrant aux valeurs extrêmes minimales, et inversement pour le maximum.
    for i in range(nb) :
        ybis[mini[i][1]] = minimum
        ybis[maxi[i][1]] = maximum
        xabmin.append(x[mini[i][1]])
        minival.append(y[mini[i][1]])
        maxival.append(y[maxi[i][1]])
        xabmax.append(x[maxi[i][1]])

    x_ab = xabmin + xabmax
    y_ab = minival + maxival
    return x_ab, y_ab, ybis

def traitement_winsorizing(y,indices_points_aberrants):
    """
    Traite les points aberrants en s'inspirant de la Winsorization : redéfini chaque point aberrant comme l'extrême non aberrant le plus proche.

    Type entrées :
        y : vecteur de float
        indices_points_aberrants : vecteur d'int

    Type sorties :
        liste modifiee

    """

    donnees = [(y[i],i) for i in range(len(y))]
    donnees.sort()

    ybis = list(y)

    if len(donnees)==len(indices_points_aberrants):
        maximum = max(y)
        minimum = min(y)
    else:

        # Recherche du minimum
        for i in range(len(donnees)):
            if donnees[i][1] not in indices_points_aberrants :
                minimum = donnees[i][0]
                break

        # Recherche du maximum
        for i in range(len(donnees)-1,-1,-1):
            if donnees[i][1] not in indices_points_aberrants :
                maximum = donnees[i][0]
                break

    for i in indices_points_aberrants :
        if ybis[i] > maximum :
            ybis[i] = maximum
        else :
            ybis[i] = minimum

    return ybis

def Faire_win(x, y, f, m_proba, M, locglob = None):

    D = {'1': ("Méthode interquartile", meth.eval_quartile),
         '2': ("Test de Chauvenet", meth.test_Chauvenet),
         '3': ("Méthode de Tau Thompson", meth.thompson),
         '4': ("Test de Grubbs", meth.grubbs),
         '5': ("Test de la déviation extreme de student", meth.deviation_extreme_student)}

    plot.scatterdata(x, y, c='bx', legend='données', new_fig=False, show=False) # affichage des points de l'échantillon

    if m_proba:
        print("Choisissez le pourcentage (< 1) de valeurs aberrantes")
        try:
            prob = float(input("> "))
        except ValueError:
            print("Valeur par défault choisie : 0.01")
            prob = 0.01
        x_ab,y_ab, ybis = winsorization(x,y,prob)
        plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon
        return ybis

    if locglob is None:
        ldt.affiche_separation()
        print("Choisir la portee de traitement des donnees :")
        print('1 : Globale')
        print('2 : Locale')
        locglob = ldt.input_choice(['1','2'])

    if M == meth.KNN:
        ldt.affiche_separation()
        print("La méthode des k plus proches voisins n'est pas compatible avec la winzorisation.")
        print("Merci de choisir une nouvelle méthode")

        for key in D:
            print(key, " : ", D[key][0])

        M = D[ldt.input_choice(list(D.keys()))][1]

    ##########################
    # Traitement des données #
    ##########################
    IND_AB = []
    if locglob == '1':
        yd, v_poids, indices_aberrants = det.supprime(y, M)
        indices_aberrants.sort()
        IND_AB = indices_aberrants
        # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
        # On ne garde que les x associés aux y.
        x_ab = []
        y_ab = []
        for i in range(len(indices_aberrants)):
            x_ab = np.append(x_ab,x[indices_aberrants[i]])
            y_ab = np.append(y_ab,y[indices_aberrants[i]])
        IND_AB = list(set(IND_AB))

        ybis = traitement_winsorizing(y, IND_AB)
        plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon
    else:
        ldt.affiche_separation()
        IND_AB = []

        ldt.affiche_separation()
        print("Quelle methode de création d'intervalles utiliser ?")
        print("1 : Par pas")
        print("2 : Par densité")
        p_meth = ldt.input_choice(['1','2'])
        if p_meth == '1':
            ep = meth.esti_epsilon(y)
            p = det.pas_inter(y,epsilon = ep) #ESSAI
        else:
            p = meth.ind_densite(y)

        if M == meth.eval_quartile :
            p = meth.regrouper(p,t=30)

        if p[-1] != len(x):
            p.append(len(x))
        print(x[-1])
        b = p[0]
        i=1
        ybis = []
        nt = 0
        while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i] #On récupère cette borne après avoir décalé
            j = x[a:b]
            g = y[a:b]
            nt+=len(g)

            _, _, indices_aberrants = det.supprime(g, M)

            ind_reels = [indice+a for indice in indices_aberrants]
            IND_AB = IND_AB + ind_reels

            ybis = ybis+traitement_winsorizing(g, indices_aberrants)

            i+=1 # On se décale d'un cran à droite
        x_ab, y_ab = [], []
        for indice in IND_AB:
            x_ab.append(x[indice])
            y_ab.append(y[indice])
        plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon
    return ybis


if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer appli_interpolaspline.")
