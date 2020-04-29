# -*- coding: utf - 8 -*-
"""
Created on Tue Apr  7 14:29:18 2020

@author: Interpolaspline
"""

# Récupération des tests par fichier ou directement des signaux
import load_tests as ldt
import plotingv2 as plot

# Methodes de detection
import winsorizing as win
import Tache_4_methodes as meth


###############################################
# Fonctions de supression de points aberrants #
###############################################


def supprime(x, methode, sup_poids=True, poids=1 / 100, k=7, m=25, y=None):  # A AJOUTER (AMELYS): OPTIONS DES METHODES
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

    if methode is meth.eval_quartile:
        a, b = meth.quartile(x)

    if methode is meth.grubbs:
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
    elif methode is meth.deviation_extreme_student:
        est_aberrant = methode(x)
        for i in range(n):
            if est_aberrant[i]:
                indices.append(i)
                if sup_poids:
                    x_sup[i] = None
                else:
                    v_poids[i] = poids

    elif methode is meth.KNN:
        ind, _, x_sup, y = meth.KNN(x, y, k, m)
        return x_sup, y

    else:

        for i in range(n):
            aberrant = False
            if methode is meth.test_Chauvenet or methode is meth.thompson:
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

def tester(x, y, f=None, M_int=None, locglob=None):
    """
    partie du programme principal :
        applique une methode de detection des points aberrants sur un ensemble de donnees

    Type des entrées :
    x et y : vecteurs de float ou vecteur d'int
    f : fonction
    M_int : caractère (compris dans : ['1', '2', '3', '4', '5', '6'])
    locglob : '1' ou '2'

    Type des sorties :
        X et Y : liste[int], liste[int]
    """

    #######################
    # Choix de la méthode #
    #######################
    if M_int is None:
        ldt.affiche_separation()
        print("Choisissez une méthode de détection des points aberrants :")
        print("1 : Inter-Quartile")
        print("2 : Test de Chauvenet")
        print("3 : Test Tau de Thompson")
        print("4 : Test de Grubbs")
        print("5 : Test de la deviation extreme de Student")
        print("6 : Test des k plus proches voisins ")

        M_int = ldt.input_choice(['1', '2', '3', '4', '5', '6'])

    D = {'1': ("Méthode interquartile", meth.eval_quartile),
         '2': ("Test de Chauvenet", meth.test_Chauvenet),
         '3': ("Test Tau de Thompson", meth.thompson),
         '4': ("Test de Grubbs", meth.grubbs),
         '5': ("Test de la déviation extreme de student", meth.deviation_extreme_student),
         '6': ("Test des k plus proches voisins", meth.KNN)}

    lab, M = D[M_int]

    if locglob is None:
        print("Choisir la portee de traitement des donnees :")
        print('1 : Globale')
        print('2 : Locale')
        locglob = ldt.input_choice(['1', '2'])

    ##########################
    # Traitement des données #
    ##########################

    if locglob == '1':
        if M is meth.KNN:
            xd, yd = supprime(x, M, y=y)
            X = xd
            Y = yd
        else:
            yd, _, indices_aberrants = supprime(y, M)
            indices_aberrants.sort()
            # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
            # On ne garde que les x associés aux y.
            xd = list(x)
            for ind in range(len(indices_aberrants) - 1, - 1, - 1):  # On part de la fin pour ne pas avoir de décalage d'indices
                xd.pop(indices_aberrants[ind])

            X = xd
            Y = yd


    else:

        ldt.affiche_separation()
        print("Quelle methode de création d'intervalles utiliser ?")
        print("1 : Par pas")
        print("2 : Par densité")
        p_meth = ldt.input_choice(['1', '2'])
        if p_meth == '1':
            ep = meth.esti_epsilon(y)
            p = pas_inter(y, epsilon=ep) #ESSAI
        else:
            p = meth.ind_densite(y)

        if M is meth.eval_quartile:
            p = meth.regrouper(p, t=30)

        if p[-1] != len(x):
            p.append(len(x))
        b = p[0]
        X = []
        Y = []
        i = 1
        while i < len(p): # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
            a = b
            b = p[i] #On récupère cette borne après avoir décalé

            j = x[a:b]
            g = y[a:b]

            if M is meth.KNN:
                k = (b - a + 1) //2
                _, _, xd, yd = meth.KNN(j, g, k, 15)
                X = X + xd
                Y = Y + yd
            else:
                yd, _, indices_aberrants = supprime(g, M)
                indices_aberrants.sort()
                # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
                # On ne garde que les x associés aux y.
                xd = list(j)
                for ind in range(len(indices_aberrants) - 1, - 1, - 1):  # On part de la fin pour ne pas avoir de décalage d'indices
                    xd.pop(indices_aberrants[ind])

                X = X + xd
                Y = Y + yd
            #x_ab, y_ab, xd, yd = meth.KNN(j, g, k, 15)



            #X = X + xd
            #Y = Y + yd

            i += 1 # On se décale d'un cran à droite

    plot.scatterdata(x, y, c='r+', legend="données", title=lab, new_fig=True, show=False)
    plot.scatterdata(X, Y, c='b+', legend='données conservées, dites "non aberrantes" ', new_fig=False, show=False)


    return X, Y


def trouve_points_aberrants():
    """
    Fonction principale : appelée depuis le menu principal
    Début de la gestion depoints aberrants
    (récupération de fichiers, choix des méthodes)

    Sorties :
        Xtab, Ytab : listes ou listes de listes (dépend du type de données)
        f : fonction (si création de signal, None sinon)
        is_array : booleen (type de Xtab et Ytab)
    """

    D_meth = {"1": "Inter-Quartile",
              "2": "Test de Chauvenet",
              "3": "Test Tau de Thompson",
              "4": "Test de Grubbs",
              "5": "Test de la deviation extreme de Student",
              "6": "Test des k plus proches voisins "}

    ldt.affiche_separation()
    print("Bienvenue dans ce gestionnaire des points aberrants !")
    x, y, f, M, is_array, seed = ldt.charge_donnees(D_meth,liste_test=True)
    if seed is not None:
        ldt.affiche_separation()
        print("Graine pour la génération du signal : ", seed)
        ldt.affiche_separation()

    ldt.affiche_separation()
    print("Supprimer les points aberrants ou les modifier ?\n")
    print("1 : Supprimer")
    print("2 : Modifier")
    sup_poids = ldt.input_choice(['1', '2'])
    ldt.affiche_separation()

    if sup_poids == "2":
        print("Choisissez la methode de modification des donnees :")
        print("1 : Winsoring (détection et traitement)")
        print("2 : Winsoring adaptée (traitement uniquement)")
        print("3 : LOESS adaptée (traitement uniquement)")
        type_mod = ldt.input_choice(['1', '2', '3'])

    if sup_poids == "2":

        D = {'1': ("Méthode interquartile", meth.eval_quartile),
             '2': ("Test de Chauvenet", meth.test_Chauvenet),
             '3': ("Méthode Tau de Thompson", meth.thompson),
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
                    print("Choisissez une méthode de détection des points aberrants :")
                    print("1 : Inter-Quartile")
                    print("2 : Test de Chauvenet")
                    print("3 : Test de Tau Thompson")
                    print("4 : Test de Grubbs")
                    print("5 : Test de la deviation extreme de Student")
                    print("6 : Test des k plus proches voisins ")

                    M_int = ldt.input_choice(['1', '2', '3', '4', '5', '6'])
                    _, Mi = D[M_int]
                else:
                    _, Mi = D[M]

                if type_mod == "3":
                    x_abi, y_abi, y_estii = meth.LOESS(xi, y[i], f, Mi)
                    x_ab.append(x_abi)
                    y_ab.append(y_abi)
                    plot.scatterdata(x, y, c='bx', legend='données', new_fig=True, show=False) # affichage des points de l'échantillon
                    plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon

                else:
                    y_estii = win.Faire_win(xi, y[i], f, type_mod == "1", Mi)

                y_esti.append(y_estii)


        else:

            if type_mod != "1":
                ldt.affiche_separation()
                print("Choisissez une méthode de détection des points aberrants :")
                print("1 : Inter-Quartile")
                print("2 : Test de Chauvenet")
                print("3 : Test Tau de Thompson")
                print("4 : Test de Grubbs")
                print("5 : Test de la deviation extreme de Student")
                print("6 : Test des k plus proches voisins ")

                M = ldt.input_choice(['1', '2', '3', '4', '5', '6'])
                _, Mi = D[M]
            else:
                Mi = None

            if type_mod == "3":
                x_ab, y_ab, y_esti = meth.LOESS(x, y, f, Mi)
                plot.scatterdata(x, y, c='bx', legend='données', new_fig=True, show=False) # affichage des points de l'échantillon
                plot.scatterdata(x_ab, y_ab, c='rx', legend='données aberrantes', new_fig=False, show=False) # affichage des points aberrants de l'échantillon

            else:
                y_esti = win.Faire_win(x, y, f, type_mod == "1", Mi)



        return x, y_esti, f, is_array
    else:

        if is_array:
            Xtab = []
            Ytab = []
            ldt.affiche_separation()
            print("Definir une portee de traitement pour tous les fichiers ? (y = oui, n = non")
            locglob_fixe = ldt.input_choice()
            if locglob_fixe == 'y':
                ldt.affiche_separation()
                print("Choisir la portee de traitement des donnees :")
                print('1 : Globale')
                print('2 : Locale')
                locglob = ldt.input_choice(['1', '2'])

            for i, exi in enumerate(x):
                if locglob_fixe == 'y':
                    X, Y = tester(exi, y[i], f, M, locglob)
                else:
                    X, Y = tester(exi, y[i], f, M)
                Xtab.append(X)
                Ytab.append(Y)

        else:
            ldt.affiche_separation()
            print("Choisir la portee de traitement des donnees :")
            print('1 : Globale')
            print('2 : Locale')
            locglob = ldt.input_choice(['1', '2'])
            Xtab, Ytab = tester(x, y, f, M, locglob)


    return Xtab, Ytab, f, is_array


if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer appli_interpolaspline.")
