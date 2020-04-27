# -*- coding: utf-8 -*-
import math

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


if __name__ == "__main__":
    y= [1,2,1,3,10,2,1,3,0,20,-15]
    inds = [4,9,10]
    ybis = traitement_winsorizing(y, inds)
    ysec = winsorization(y,0.3)
    print(y,ybis,ysec,sep="\n")
