# -*- coding: utf-8 -*-
import math

def winsorization(y,quantite_aberrante):
    """ 
    Applique la Winsorization de manière rigoureuse, selon la définition.
    quantite_aberrante appartient à [0,1[.
    Cette fonction modifie y.
    
    Type entrées :
        y : vecteur de float
        quantite_aberrante : float
        
    Type sorties :
        aucune sortie
    """
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
        y[mini[i][1]] = minimum
        y[maxi[i][1]] = maximum    
    
    
def traitement_winsorizing(y,indices_points_aberrants):
    """
    Traite les points aberrants en s'inspirant de la Winsorization : redéfini chaque point aberrant comme l'extrême non aberrant le plus proche.
    Modifie les données.
    
    Type entrées :
        y : vecteur de float
        indices_points_aberrants : vecteur d'int
        
    Type sorties :
        aucune sortie
    
    """
    
    donnees = [(y[i],i) for i in range(len(y))]
    donnees.sort()

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
        if y[i] > maximum :
            y[i] = maximum
        else :
            y[i] = minimum
    