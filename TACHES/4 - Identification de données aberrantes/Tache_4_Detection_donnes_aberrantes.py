# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:29:18 2020

@author: Interpolaspline
"""
#gestionnaire de programme
import sys

# Récupération des tests par fichier ou directement des signaux
import load_tests as ldt
import signaux_splines as ss
import plotingv2 as plot
import numpy as np

# Methodes de detection
import Tache_4_methodes as meth


###############################################
# Fonctions de supression de points aberrants #
###############################################


def supprime(x, methode, sup_poids=True, poids=1 / 100,k=7,m=25):  # A AJOUTER (AMELYS) : OPTIONS DES METHODES
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

    else:

        for i in range(n):
            aberrant = False
            if methode == meth.test_Chauvenet or methode == meth.thompson:
                if methode(x, i):
                    aberrant = True
            elif methode == meth.KNN :
                if meth.KNN(x,i,k,m):
                    aberrant = True
            else:  # methode == eval_quartile:
                if meth.eval_quartile(x, i, a, b):
                    aberrant = True

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

def tester(x,y,f = None,M_int = None):
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
        print("6 : Test des k plus proches voisins")
        
        M_int = ldt.input_choice(['1','2','3','4','5','6'])

    D = {'1': ("Méthode interquartile", meth.eval_quartile),
         '2': ("Test de Chauvenet", meth.test_Chauvenet),
         '3': ("Méthode de Tau Thompson", meth.thompson),
         '4': ("Test de Grubbs", meth.grubbs),
         '5': ("Test de la déviation extreme de student", meth.deviation_extreme_student),
         '6': ("Test des k plus proches voisins", meth.KNN)}

    lab,M = D[M_int]
    
    ##########################
    # Traitement des données #
    ##########################
    
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

        yd, v_poids, indices_aberrants = supprime(g, M)  # AMELYS : IL FAUT GERER LE CAS Où ON NE SUPPRIME PAS LES POIDS
        indices_aberrants.sort()
        # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
        # On ne garde que les x associés aux y.
        xd = list(j)
        for ind in range(len(indices_aberrants) - 1, -1,-1):  # On part de la fin pour ne pas avoir de décalage d'indices
            xd.pop(indices_aberrants[ind])

        X = X + xd
        Y = Y + yd

        i += 1  # On se décale d'un cran à droite
        
    plot.scatterdata(x,y,c='b+',legend = "données",title = lab,new_fig = True,show = False)
    plot.scatterdata(X,Y,c='r+',legend='données conservées, dites "non aberrantes" ',new_fig = False,show = False)
    
    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi,f(xi),new_fig = False,c = 'g')
        
    plot.show()

if __name__ == "__main__":
    
    ldt.affiche_separation()
    print("Bienvenue dans ce gestionnaire des points aberrants !")
    print("Choisissez une option de récupération de données :")
    print("1 : Fichier contenant une liste de plusieurs fichiers à tester")
    print("2 : Récupération sur un fichier")
    print("3 : Générer un test")
    print("\nPour Quitter le programme, appuyer sur q lors d'un choix.")
    
    type_test = ldt.input_choice(['1','2','3'])
    
    if type_test == 'q':
        sys.exit(0)
        
    type_test = int(type_test)
    
    if type_test == 1:
        
        ldt.affiche_separation()
        f_liste_nom = input("Entrez le nom du fichier contenant la liste des tests :\n> ")
        
        if f_liste_nom == 'q':
            sys.exit(0)
            
        try :
            f_liste = open(f_liste_nom,'r')
        except :
            print("Erreur, le fichier " + f_liste_nom + " est introuvable, merci de relancer le programme.")
            sys.exit(0)
        
        ldt.affiche_separation()
        print("Définir une methode pour tous les fichiers ? (y = oui, n = non)")
        def_M = ldt.input_choice()
        
        if def_M == 'y':
            ldt.affiche_separation()
            print("Choisissez une méthode de traitement des points aberrants :")
            print("1 : Inter-Quartile")
            print("2 : Test de Chauvenet")
            print("3 : Test de Tau Thompson")
            print("4 : Test de Grubbs")
            print("5 : Test de la deviation extreme de Student")
            print("6 : Test des k plus proches voisins")
            
            M_int = ldt.input_choice(['1','2','3','4','5','6'])
        else :
            M_int = None
        
        liste =(f_liste.read()).split("\n")
        
        for f_test in liste:
            x,y = ldt.load_points(f_test)
            tester(x,y,M_int = M_int)
        
    elif type_test == 2:
        
        ldt.affiche_separation()
        f_test = input("Entrez le nom du fichier de test :\n> ")
        if f_test == 'q':
            sys.exit(0)
        x,y = ldt.load_points(f_test)
        tester(x,y)
        
    else:
        ldt.affiche_separation()
        print("Test sur génération de signal. Signal stationnaire ? (y = oui, n = non)")
        stationnaire = ldt.input_choice()
        
        # signaux de tests stationnaires provenant du générateur
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        
        if stationnaire == 'y':
            x,y, f = ss.stationary_signal((30,), 0.9, noise_func=nfunc,seed=0)
            #x,y, f = ss.stationary_signal((30,), 0.5, noise_func=nfunc)
        else: 
            x, y, f = ss.non_stationary_signal((30,), switch_prob=0.1, noise_func=nfunc)
            #x, y, f = ss.non_stationary_signal((30,), switch_prob=0.2, noise_func=nfunc)
            
        tester(x,y,f)
            
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


    
