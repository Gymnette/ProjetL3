# -*- coding: utf - 8 -*-
"""
Created on Tue Apr  7 14:29:18 2020

@author: Interpolaspline
"""

# Récupération des tests par fichier ou directement des signaux
import load_tests as ldt
import plotingv2 as plot
import numpy as np

# Methodes de detection
import Tache_4_methodes as meth


###############################################
# Fonctions de supression de points aberrants #
###############################################


def supprime(x, methode, sup_poids=True, poids=1 / 100, k=7, m=25, y = None):  # A AJOUTER (AMELYS): OPTIONS DES METHODES
    """
    Parcours toutes les valeurs de x afin de toutes les traiter.
    La fonction supprime prend un vecteur x d'ordonnées de points, une methode de
    detection des points aberrants, un booléen sup_poids égal à True si on veut supprimer
    les points aberrants, et égal à False si on veut affecter le poids "poids" aux points
    aberrants et un poids = 1 aux points considérés comme adaptés.
    Elle renvoie une liste d'ordonnées ne contenant pas celles supprimées,
    une liste de poids, ainsi qu'une liste des indices dans le vecteur des valeurs supprimées.

    type des entrees :
        x : vecteur de float ou vecteur d'int
        methode : fonction :vecteur de float ou vecteur d'int -> (float,float)
        sup_poids : booleen
        poids : float

    type des sorties : tuple (x_sup,v_poids,indices)
        x_sup : vecteur de float ou vecteur d'int
        v_poids : vecteur de float
        indices : vecteur d'int
    """
    n = len(x)
    x_sup = list(x)
    v_poids = [1] * n
    indices = []

    if methode == meth.eval_quartile:
        a, b = meth.quartile(x)

    if methode == meth.grubbs:
        res, ind = meth.grubbs(x)
        x_cpy = list(x)
        while (ind >= 0 and res):  # Un point aberrant a été trouvé de manière "classique".
            ind_reel = meth.calcul_reel(ind, indices)
            indices.append(ind_reel)
            if sup_poids:
                x_sup[ind_reel] = None
            else:
                v_poids[ind_reel] = poids

            x_cpy.pop(ind)  # c'est bien ici le relatif
            res, ind = meth.grubbs(x_cpy)
        # Si c'est res qui est faux, pas de soucis, on a notre résultat.
        # Si l'indice est négatif, le résultat sera faux, donc c'est bon, pas de point aberrant détecté.
    elif methode == meth.deviation_extreme_student:
        est_aberrant = methode(x)
        for i in range(n):
            if est_aberrant[i]:
                indices.append(i)
                if sup_poids:
                    x_sup[i] = None
                else:
                    v_poids[i] = poids

    elif methode == meth.KNN:
        ind, uk_ya, x_sup, y = meth.KNN(x, y, k, m)
        return x_sup, y

    else:

        for i in range(n):
            aberrant = False
            if methode == meth.test_Chauvenet or methode == meth.thompson:
                aberrant = methode(x, i)
            else:  # methode == eval_quartile:
                aberrant = meth.eval_quartile(x, i, a, b)

            if aberrant:
                indices.append(i)
                if sup_poids:
                    x_sup[i] = None
                else:
                    v_poids[i] = poids

    while None in x_sup:
        x_sup.remove(None)

    return x_sup, v_poids, indices




###################################
# Gestion des intervalles d'étude #
###################################

def pas_inter(y, epsilon=0.1):
    """
    Cette fonction prend un vecteur y et un paramètre de variation epsilon,
    et renvoie des intervalles sur lesquels la variation de y est inferieure à epsilon.
    Les intervalles sont représentés par une liste d'entiers, dont l'ordre est important :
    chaque entier représente un intervalle en indiquant le début de celui ci, excepté la dernière valeur indiquant la fin du dernier, exclue.
    L'intervalle représenté à l'indice i est donc [p[i],p[i+1][

    Type des entrées :
        y : vecteur de float ou vecteur d'int
        epsilon : float

    Type des sorties :
        liste[int]
    """
    p = [0]
    n = len(y)
    for i in range(n - 2):
        d_yi = y[i + 1] - y[i]
        d_yi_1 = y[i + 2] - y[i + 1]
        delta = abs(d_yi - d_yi_1)
        if delta > epsilon:
            p.append(i + 1)

    # Les deux derniers points appartiendront toujours au dernier intervalle.
    p.append(n)

    return p

def tester(x, y, f = None, M_int = None, locglob='1'):
    """
    partie du programme principal :
        applique une methode de detection des points aberrants sur un ensemble de donnees
    """

    #######################
    # Choix de la méthode #
    #######################
    if M_int is None:
        ldt.affiche_separation()
        print("Choisissez une méthode de traitement des points aberrants :")
        print("1 : Inter-Quartile")
        print("2 : Test de Chauvenet")
        print("3 : Test de Tau Thompson")
        print("4 : Test de Grubbs")
        print("5 : Test de la deviation extreme de Student")
        print("6 : Test des k plus proches voisins ")

        M_int = ldt.input_choice(['1', '2', '3', '4', '5', '6'])

    D = {'1': ("Méthode interquartile", meth.eval_quartile),
         '2': ("Test de Chauvenet", meth.test_Chauvenet),
         '3': ("Méthode de Tau Thompson", meth.thompson),
         '4': ("Test de Grubbs", meth.grubbs),
         '5': ("Test de la déviation extreme de student", meth.deviation_extreme_student),
         '6': ("Test des k plus proches voisins", meth.KNN)}

    lab, M = D[M_int]

    ##########################
    # Traitement des données #
    ##########################
    if locglob == '1':
        p = pas_inter(y, epsilon=0.5)
        b = p[0]
        X = []
        Y = []
        i = 1
        while i < len(p):  # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i]  # On récupère cette borne après avoir décalé

            j = x[a:b]
            g = y[a:b]

            if M == meth.KNN:
                xd, yd = supprime(j, M, y = g)
                X = X + xd
                Y = Y + yd
            else:
                yd, v_poids, indices_aberrants = supprime(g, M)  # AMELYS: IL FAUT GERER LE CAS Où ON NE SUPPRIME PAS LES POIDS
                indices_aberrants.sort()
                # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
                # On ne garde que les x associés aux y.
                xd = list(j)
                for ind in range(len(indices_aberrants) - 1, - 1, - 1):  # On part de la fin pour ne pas avoir de décalage d'indices
                    xd.pop(indices_aberrants[ind])

                X = X + xd
                Y = Y + yd

            i += 1  # On se décale d'un cran à droite

    else:

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
        X = []
        Y = []
        i=1
        while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i] #On récupère cette borne après avoir décalé

            j = x[a:b+1]
            g = y[a:b+1]
            k = (b-a+1)//2

            x_ab, y_ab,xd, yd = meth.KNN(j,g,k,15) #AMELYS : IL FAUT GERER LE CAS Où ON NE SUPPRIME PAS LES POIDS



            X = X + xd
            Y = Y + yd

            i+=1 # On se décale d'un cran à droite

    plot.scatterdata(x, y, c='b+', legend = "données", title = lab, new_fig = True, show = False)
    plot.scatterdata(X, Y, c='r+', legend='données conservées, dites "non aberrantes" ', new_fig = False, show = False)

    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi, f(xi), new_fig = False, c = 'g')

    plot.show()
    return X, Y


def trouve_points_aberrants():

    D_meth = {"1": "Inter-Quartile",
            "2": "Test de Chauvenet",
            "3": "Test de Tau Thompson",
            "4": "Test de Grubbs",
            "5": "Test de la deviation extreme de Student",
            "6": "Test des k plus proches voisins "}

    ldt.affiche_separation()
    print("Bienvenue dans ce gestionnaire des points aberrants !")
    x, y, f, M, is_array, seed = ldt.charge_donnees(D_meth)
    if seed is not None:
        ldt.affiche_separation()
        print("Graine pour la génération du signal : ", seed)
        ldt.affiche_separation()

    ldt.affiche_separation()
    print("Supprimer les points aberrants ou les modifier ?\n")
    print("1 : Supprimer")
    print("2 : Modifier")
    sup_poids = ldt.input_choice(['1','2'])
    ldt.affiche_separation()

    if sup_poids == "2":

        D = {'1': ("Méthode interquartile", meth.eval_quartile),
             '2': ("Test de Chauvenet", meth.test_Chauvenet),
             '3': ("Méthode de Tau Thompson", meth.thompson),
             '4': ("Test de Grubbs", meth.grubbs),
             '5': ("Test de la déviation extreme de student", meth.deviation_extreme_student),
             '6': ("Test des k plus proches voisins", meth.KNN)}

        if is_array:

            x_ab = []
            y_ab = []
            y_esti = []

            for i, xi in enumerate(x):
                if M is None:
                    ldt.affiche_separation()
                    print("Choisissez une méthode de traitement des points aberrants :")
                    print("1 : Inter-Quartile")
                    print("2 : Test de Chauvenet")
                    print("3 : Test de Tau Thompson")
                    print("4 : Test de Grubbs")
                    print("5 : Test de la deviation extreme de Student")
                    print("6 : Test des k plus proches voisins ")

                    M_int = ldt.input_choice(['1', '2', '3', '4', '5', '6'])
                    lab, Mi = D[M_int]
                else:
                    lab, Mi = D[M]
                x_abi, y_abi, y_estii = meth.LOESS(xi, y[i], f, Mi)
                x_ab.append(x_abi)
                y_ab.append(y_abi)
                y_esti.append(y_estii)
                plot.scatterdata(xi, y[i], c='bx', legend='données',show=False) # affichage des points de l'échantillon
                plot.scatterdata(x_abi, y_abi, c='rx', legend='données aberrantes', new_fig=False) # affichage des points aberrants de l'échantillon
               

        else:

            ldt.affiche_separation()
            print("Choisissez une méthode de traitement des points aberrants :")
            print("1 : Inter-Quartile")
            print("2 : Test de Chauvenet")
            print("3 : Test de Tau Thompson")
            print("4 : Test de Grubbs")
            print("5 : Test de la deviation extreme de Student")
            print("6 : Test des k plus proches voisins ")

            M = ldt.input_choice(['1', '2', '3', '4', '5', '6'])
            lab, Mi = D[M]
            x_ab, y_ab, y_esti = meth.LOESS(x, y, f, Mi)

            plot.scatterdata(x, y, c='bx', legend='données',show=False) # affichage des points de l'échantillon
            plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False) # affichage des points aberrants de l'échantillon
           

        return x, y_esti, f, is_array
    else :

        if is_array:
            Xtab = []
            Ytab = []
            for i, exi in enumerate(x):
                X, Y = tester(exi, y[i], f, M)
                Xtab.append(X)
                Ytab.append(Y)

        else:
            ldt.affiche_separation()
            print("Choisir la portee de traitement des donnees :")
            print('1 : Local')
            print('2 : Global')
            locglob = ldt.input_choice(['1','2'])
            Xtab, Ytab = tester(x, y, f, M, locglob)


    return Xtab, Ytab, f, is_array

        #############################################################
        # Epsilon à choisir en fonction des graines et des méthodes #
        #############################################################
        # Pour les signaux stationnaires de paramètres 30, et 0.9
        # Pour les paramètres des méthodes par défaut
        #           0      1       2       3       4        5
        # Quartile  0.5
        # Chauvenet
        # Thompson
        # Grubbs    0.3
        # ESD       0.3

if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")

