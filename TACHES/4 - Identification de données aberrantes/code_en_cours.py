# -*- coding: utf-8 -*-
import load_tests as ldt


#############################################
# Méthodes de détection de points aberrants #
#############################################

def quartile(x):
    """
    Méthode inter-quartile.
    La fonction prend une liste de valeurs (ordonnées de points) et renvoie un intervalle [a,b] associé.
    Un point sera considéré comme aberrant si son ordonnée n'appartient pas à l'intervalle [a,b]
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        
    type des sorties :
        (float,float)
        
    """
    n = len(x)
    k = n//4 if n%4 == 0 else n//4+1
    # le premier quartile
    q1 = x[k-1]
    # le 3éme quartile
    q3 = x[3*k-1]
    # l'inter-quartile
    inter_q = q3-q1
    
    return q1-1.5*inter_q, q3+1.5*inter_q



###############################################
# Fonctions de supression de points aberrants #
###############################################

def supprime(x,methode,sup_poids= True,poids=1/100):
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
    a,b = methode(x)
    x_sup = list(x)
    v_poids = [1]*n
    indices = []
    for i,e in enumerate(x):
        if e <a or e>b:
            indices.append(i)
            if sup_poids:
                x_sup[i] = None
            else :
                v_poids[i] = poids
    
    while None in x_sup:
        x_sup.remove(None)
    
    return x_sup,v_poids,indices
            

def supprime_un(x,v_poids,i,methode,sup_poids= 2,poids=1/100):
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
    
    a,b = methode(x)
    xk = x[i]
    x_sup = list(x)
    if xk <a or xk>b:
        if sup_poids == 1:
            x_sup.pop(i)
            v_poids.pop(i)
        elif sup_poids == 2:
            x_sup[i] = None
            v_poids[i] = None
        elif sup_poids == 3:
            x_sup[i] = a if xk<a else b
        else:
            v_poids[i] = poids

    return x_sup,v_poids

if __name__ == "__main__":
    X,Y = ldt.load_points("test.txt")

    
