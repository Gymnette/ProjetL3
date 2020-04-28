# -*- coding: utf - 8 -*-
"""
Avril 2020

@author: Interpolaspline

"""
import sys
import numpy as np

import signaux_splines as ss


def affiche_liste_fichiers_liste():
    affiche_separation()
    print("Pour le fichier contenant tous les fichiers du repertoire Test,")
    print("entrez Tous_les_tests")


def affiche_noms_fichiers_tests():
    """
    Affiche les fichiers de Tests\
    """
    os = sys.platform
    sep = '\\' if os == 'win32' else '/'
    try:
        f_liste = open("Tests"+sep+"Tous_les_tests.txt", 'r')
    except IOError:
        print("Erreur, le fichier Tests"+sep+"Tous_les_tests.txt est introuvable, merci de relancer le programme.")
        sys.exit(0)

    liste = (f_liste.read()).split("\n")
    affiche_separation()
    print("liste des fichiers de test disponibles :")
    for f in liste:
        print(f)


def load_points(fichier):
    """
    Recupere les vecteurs X,Y dans un fichier donne
    """
    try:
        (X, Y) = np.loadtxt(fichier)
    except IOError:
        print("Erreur, le fichier " + fichier + " est introuvable, merci de relancer le programme.")
        sys.exit(0)

    return X, Y


def sortpoints(X, Y):
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
    return Xb, Yb

def remplissent_conditions(x,y):
    """
    Teste l'integrite des donnees avant interpolation'
    """
    ok = True
    err = None

    if not len(x) == len(y):
        ok = False
        err = "ERREUR : Le vecteur des abscisses et celui des ordonnees n'ont pas la meme taille"
    if len(x) == 0 :
        ok = False
        err = "ERREUR : Donnees vides"

    return ok,err

def input_choice(Choices=None):
    """
    Type de Choices : Liste des valeurs d'input possible
    """
    if Choices is None:
        Choices = ['y', 'n']
    Choices.append('q')
    ipt = input("> ")
    while ipt not in Choices:
        print("Merci d'entrer", end=" ")
        for i in range(len(Choices) - 1):
            print(Choices[i] + ",", end=" ")
        print(" ou ", Choices[-1])
        ipt = input("> ")
    if ipt == 'q':
        sys.exit(0)
    return ipt

def affiche_separation(c='-', n=50):
    """
    affiche n fois le caractere c
    """
    print(c * n)


def charge_methodes(D_methodes=None, une_methode=False):
    """
    Charge methode pouvoir appliquer les methodes
    """

    if D_methodes is not None:
        if not une_methode:
            affiche_separation()
            print("\nDéfinir une methode pour toutes les données ? (y = oui, n = non)")
            def_M = input_choice()
        else:
            def_M = 'y'

        if def_M == 'y':

            affiche_separation()
            print("\nChoisissez le mode de traitement des données :")
            for key in D_methodes.keys():
                print(key, " : ", D_methodes[key])

            M_int = input_choice(list(D_methodes.keys()))
        else:
            M_int = None
    else:
        M_int = None
    return M_int


def charge_donnees(D_methodes=None):
    """
    Charge x,y, fonction, methode, et un booleen qui donne le type de x et y pour pouvoiir appliquer les methodes
    """
    print("Choisissez une option de récupération de données :")
    print("1 : Fichier contenant une liste de plusieurs fichiers à tester")
    print("2 : Récupération sur un fichier")
    print("3 : Générer un test")
    print("4 : Recréer un test à partir d'une graine")

    type_test = input_choice(['1', '2', '3', '4'])

    type_test = int(type_test)

    os = sys.platform
    sep = '\\' if os == 'win32' else '/'

    if type_test == 1:

        affiche_liste_fichiers_liste()
        affiche_separation()
        f_liste_nom = "Tests"+sep+input("Entrez le nom du fichier contenant la liste des tests :\n> ")+".txt"

        if f_liste_nom == 'q':
            sys.exit(0)

        try:
            f_liste = open(f_liste_nom, 'r')
        except IOError:
            print("Erreur, le fichier " + f_liste_nom + " est introuvable, merci de relancer le programme.")
            sys.exit(0)

        if D_methodes is not None:
            affiche_separation()
            print("\nDéfinir une methode pour tous les fichiers ? (y = oui, n = non)")
            def_M = input_choice()

            if def_M == 'y':

                affiche_separation()
                print("\nChoisissez le mode de traitement du fichier :")
                for key in D_methodes.keys():
                    print(key, " : ", D_methodes[key])

                M_int = input_choice(list(D_methodes.keys()))
            else:
                M_int = None
        else:
            M_int = None

        liste = (f_liste.read()).split("\n")

        X = []
        Y = []
        for f_test in liste:
            x, y = load_points("Tests"+sep+f_test+".txt")
            X.append(x)
            Y.append(y)
        return X, Y, None, M_int, True, None

    if type_test == 2:

        affiche_noms_fichiers_tests()
        affiche_separation()
        f_test = "Tests"+sep+input("Entrez le nom du fichier de test :\n> ")+".txt"
        if f_test == 'q':
            sys.exit(0)
        x, y = load_points(f_test)
        return x, y, None, None, False, None

    if type_test == 3:
        affiche_separation()
        print("\nTest sur génération de signal. Signal stationnaire ? (y = oui, n = non)")
        stationnaire = input_choice()

        std1 = 0.05
        nb_pts = 30
        prob = 0.1
        reg = 0.1

        affiche_separation()
        print("\nCreation de signal rapide ? (y = oui, n = non)")
        print("Les paramètres seraient :")
        print("- Ecart type = ",std1)
        print("- Nombre de points = ",nb_pts)
        if stationnaire == 'y':
            print("- Régularité = ",reg)
        else:
            print("- Probabilité de saut = ",prob)

        skip = input_choice()

        # signaux de tests stationnaires provenant du générateur
        if skip == 'n':
            print("Choisissez les valeurs suivantes :")
            std1 = float(input("(flottant) Ecart type = "))
            nb_pts = int(input("(entier) Nombre de points = "))
            if stationnaire == 'n':
                prob = float(input("(flottant) Probabilité de saut = "))
            else:
                reg = float(input("(flottant [0,1]) Régularité = "))
        nfunc = lambda x: ss.add_bivariate_noise(x, std1)

        if stationnaire == 'y':
            x, y, f, seed = ss.stationary_signal((nb_pts, ), regularity=reg, noise_func=nfunc)
        else:
            x, y, f, seed = ss.non_stationary_signal((nb_pts, ), switch_prob=prob, noise_func=nfunc)

        return x, y, f, None, False, seed

    std1 = 0.05
    nb_pts = 30
    prob = 0.1
    reg = 0.1
    affiche_separation()
    print("\nTest sur recréation de signal. Signal stationnaire ? (y = oui, n = non)")
    stationnaire = input_choice()

    affiche_separation()
    print("\nQuelle graine utiliser ?")
    seed = - 1
    while seed < 0:
        try:
            seed = int(input("> "))
            if seed < 0 or seed >= 2 ** 32 -1:
                print("Merci d'entrer un nombre valide")
        except ValueError:
            print("Merci d'entrer un nombre valide")
            seed = - 1
    # signaux de tests stationnaires provenant du générateur
    print("\nCreation de signal rapide ? (y = oui, n = non)")
    print("Les paramètres seraient :")
    print("- Ecart type = ",std1)
    print("- Nombre de points = ",nb_pts)
    if stationnaire == 'y':
        print("- Régularité = ",reg)
    else:
        print("- Probabilité de saut = ",prob)
    skip = input_choice()

    # signaux de tests stationnaires provenant du générateur
    if skip == 'y':
        std1 = 0.05
        nb_pts = 30
        prob = 0.1
        reg = 0.1
    else:
        print("Choisissez les valeurs suivantes :")
        std1 = float(input("(flottant) Ecart type = "))
        nb_pts = int(input("(entier) Nombre de points = "))
        if stationnaire == 'n':
            prob = float(input("(flottant) probabilité de saut = "))
        else:
            reg = float(input("(flottant [0,1]) régularité = "))
    nfunc = lambda x: ss.add_bivariate_noise(x, std1)

    if stationnaire == 'y':
        x, y, f, seed = ss.stationary_signal((nb_pts, ), regularity=reg, noise_func=nfunc, seed=seed)
    else:
        x, y, f, seed = ss.non_stationary_signal((nb_pts, ), switch_prob=prob, noise_func=nfunc, seed=seed)

    return x, y, f, None, False, seed

if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")
