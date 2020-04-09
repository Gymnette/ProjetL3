# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:55:51 2020

@author: amelys
"""
def pas_inter_essai(y, epsilon=0.1):
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
        d_yi = abs(y[i + 1] - y[i])
        d_yi_1 = abs(y[i + 2] - y[i])
        print(d_yi, d_yi_1, i)

        if (d_yi > epsilon and d_yi_1 > epsilon):
            print(i)
            p.append(i + 1)
        if d_yi > epsilon and d_yi_1 <= epsilon:
            i += 1  # Il y a eu un point "aberrant"(c'est bête de ne pas le retirer tout de suite...)

    # Les deux derniers points appartiendront toujours au dernier intervalle.
    p.append(n)

    return p

def supprime_un(x, v_poids, i, methode, sup_poids=2,
                poids=1 / 100):  # COMMENTAIRE BERYL : PAS TOUCHEE, JE TE LE LAISSE AMELYS
    """
    Traite une valeur de x, donnée par l'indice i.
    La fonction supprime prend un vecteur x d'ordonnées, le vecteur des poids associés,
    un indice i de l'élément à supprimer, une méthode de détection des points aberrants, u
    n entier sup_poids égal à 1 si on veut supprimer les points aberrants, 
    égal à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
    égal à 3 si on veut remplacer les points aberrants par les valeurs non aberrantes les plus proches (Méthode de Winsorising) :
        - Affecte le quartile le plus proche pour la méthode interquartile
        DECRIRE ICI CE QUI EST FAIT POUR LES AUTRES METHOOOOOOOOOOOOOOOODES
    et égal à 0 si on veut affecter le poids "poids" aux points aberrants et un poids = 1 aux points normaux.
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        v_poids : vecteur de float
        methode : fonction : vecteur de float ou vecteur d'int -> (float, float)
        sup_poid : 0,1,2,3
        poids : float
        
    type des sorties : couple (x_sup, v_poids)
        x_sup : vecteur de float ou vecteur d'int
        v_poids : vecteur de float
    
    """

    a, b = methode(x)
    xk = x[i]
    x_sup = list(x)
    if xk < a or xk > b:
        if sup_poids == 1:
            x_sup.pop(i)
            v_poids.pop(i)
        elif sup_poids == 2:
            x_sup[i] = None
            v_poids[i] = None
        elif sup_poids == 3:
            x_sup[i] = a if xk < a else b
        else:
            v_poids[i] = poids

    return x_sup, v_poids

