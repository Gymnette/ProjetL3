# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:09:26 2020

@author: amelys
"""
import sys
import numpy as np

import signaux_splines as ss

def load_points(fichier):
    try :
        (X,Y) = np.loadtxt(fichier)
    except :
        print("Erreur, le fichier " + fichier + " est introuvable, merci de relancer le programme.")
        sys.exit(0)

    return X,Y
    
def sortpoints(X,Y):
    """
    Fonction de tri de deux listes X et Y en fonction de X.
    Trie la liste X (sans modification globale) et range Y pour que son ordre corresponde
    au tri de X.
    
    Exemples :
    sortpoints([1,3,2,4],[5,6,7,8]) = [1,2,3,4],[5,7,6,8]
    """
    D = {}
    n = len(X)
    Xb = list(X)
    for i in range(n):
        D[X[i]] = Y[i]
    Xb.sort()
    Yb = [D[Xb[i]] for i in range(n)]
    return Xb,Yb

def input_choice(Choices = ['y','n']):
    """
    Type de Choices : Liste des valeurs d'input possible
    """
    Choices.append('q')
    ipt = input("> ")
    while ipt not in Choices:
        print("Merci d'entrer", end = " ")
        for i in range(len(Choices)-1):
            print(Choices[i] + ",",end = " ")
        print(" ou ",Choices[-1])
        ipt = input("> ")
    if ipt == 'q':
        sys.exit(0)
    return ipt

def affiche_separation(c = '-',n = 50):
    print(c*n)
    
def charge_donnees(D_methodes=None):
    """
    Charge x,y, fonction, methode, et un booleen qui donne le type de x et y pour pouvoiir appliquer les methodes
    """
    print("Choisissez une option de récupération de données :")
    print("1 : Fichier contenant une liste de plusieurs fichiers à tester")
    print("2 : Récupération sur un fichier")
    print("3 : Générer un test")
    
    type_test = input_choice(['1','2','3'])
    
    type_test = int(type_test)
    
    if type_test == 1:
        
        affiche_separation()
        f_liste_nom = input("Entrez le nom du fichier contenant la liste des tests :\n> ")
        
        if f_liste_nom == 'q':
            sys.exit(0)
            
        try :
            f_liste = open(f_liste_nom,'r')
        except :
            print("Erreur, le fichier " + f_liste_nom + " est introuvable, merci de relancer le programme.")
            sys.exit(0)
        
        if D_methodes is not None:
            affiche_separation()
            print("Définir une methode pour tous les fichiers ? (y = oui, n = non)")
            def_M = input_choice()
            
            if def_M == 'y':
                
                affiche_separation()
                print("Choisissez le mode de traitement du fichier :")
                for key in D_methodes.keys():
                    print(key," : ",D_methodes[key])
                     
                M_int = input_choice(list(D_methodes.keys()))
            else :
                M_int = None
        else:
            M_int = None
        
        liste =(f_liste.read()).split("\n")
        
        X = []
        Y = []
        for f_test in liste:
            x,y = load_points(f_test)
            X.append(x)
            Y.append(y)
        return X,Y,None,M_int,True
        
    elif type_test == 2:
        
        affiche_separation()
        f_test = input("Entrez le nom du fichier de test :\n> ")
        if f_test == 'q':
            sys.exit(0)
        x,y = load_points(f_test)
        return x,y,None,None,False
        
    else:
        affiche_separation()
        print("\nTest sur génération de signal. Signal stationnaire ? (y = oui, n = non)")
        stationnaire = input_choice()
        
        # signaux de tests stationnaires provenant du générateur
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        
        if stationnaire == 'y':
            x,y, f = ss.stationary_signal((30,), 0.9, noise_func=nfunc,seed=0)
            #x,y, f = ss.stationary_signal((30,), 0.5, noise_func=nfunc)
        else: 
            x, y, f = ss.non_stationary_signal((30,), switch_prob=0.1, noise_func=nfunc)
            #x, y, f = ss.non_stationary_signal((30,), switch_prob=0.2, noise_func=nfunc)
            
        return x,y,f,None,False
            
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