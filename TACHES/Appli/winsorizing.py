# -*- coding: utf-8 -*-
import math
import Tache_4_Detection_donnes_aberrantes as det
import load_tests as ldt
import Tache_4_methodes as meth
import plotingv2 as plot
import numpy as np
def winsorization(y,quantite_aberrante):
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

    return ybis

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
    
    
    plot.scatterdata(x, y, c='bx', legend='données', new_fig=False, show=False) # affichage des points de l'échantillon
    
    if locglob is None:
        print("Choisir la portee de traitement des donnees :")
        print('1 : Global')
        print('2 : Local')
        locglob = ldt.input_choice(['1','2'])
        
    if M == meth.KNN:
        M = meth.eval_quartile


    ##########################
    # Traitement des données #
    ##########################
    IND_AB = []
    if locglob == '1':
        p = det.pas_inter(y, epsilon=0.5)
        b = p[0]
        i = 1
        while i < len(p):  # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i]  # On récupère cette borne après avoir décalé

            j = x[a:b]
            g = y[a:b]

            _, _, indices_aberrants = det.supprime(g, M)  
            indices_aberrants.sort()
            IND_AB = IND_AB + indices_aberrants
            # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
            # On ne garde que les x associés aux y.
            x_ab = []
            y_ab = []
                
            xd = list(j)
            for ind in range(len(indices_aberrants) - 1, - 1, - 1):  # On part de la fin pour ne pas avoir de décalage d'indices
                xd.pop(indices_aberrants[ind])

            i += 1  # On se décale d'un cran à droite
        for i in range(len(indices_aberrants)):
                x_ab = np.append(x_ab,x[indices_aberrants[i]])
                y_ab = np.append(y_ab,y[indices_aberrants[i]])
        plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon
    else:
        ldt.affiche_separation()
        IND_AB = []

        ldt.affiche_separation()
        print("Quelle methode de création d'intervalles utiliser ?")
        print("1 : Par ???????")
        print("2 : Par densité")
        p_meth = ldt.input_choice(['1','2'])
        if p_meth == '1':
            ep = meth.esti_epsilon(y)
            p = pas_inter(y,epsilon = ep) #ESSAI
        else:
            p = meth.ind_densite(y)
        p = meth.regrouper(p,30)

        b = p[0]
        i=1
        while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i] #On récupère cette borne après avoir décalé

            j = x[a:b+1]
            g = y[a:b+1]
            k = (b-a+1)//2

            _, _, indices_aberrants = det.supprime(g, M)
            IND_AB = IND_AB + indices_aberrants

            i+=1 # On se décale d'un cran à droite

    IND_AB = list(set(IND_AB))
    
    if m_proba:
        prob = len(IND_AB)/len(y)
        ybis = winsorization(y,prob)
    else:
        ybis = traitement_winsorizing(y, IND_AB)

    return ybis



if __name__ == "__main__":
    y= [1,2,1,3,10,2,1,3,0,20,-15]
    inds = [4,9,10]
    ybis = traitement_winsorizing(y, inds)
    ysec = winsorization(y,0.3)
    print(y,ybis,ysec,sep="\n")
