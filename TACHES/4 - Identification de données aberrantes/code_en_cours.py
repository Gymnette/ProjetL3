# -*- coding: utf-8 -*-


import load_tests as ldt


def quartile(x):
    """
    la fonction prend un vecteur et qui renvoie un intervalle [a,b]
    un point est considéré aberrant s'il n'appartient pas à intervalle [a,b]
    
    entrees :
        x : vecteur trie
        
    sorties :
        [a,b] : un intervalle (float,float)
        
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

def supprime(x,methode,sup_poid= True,poids=1/100):
    """
    la fonction supprime prend un vecteur x, le nom une methode de
    detection des points aberrants, un booléen sup_poid égale à True si on veut supprimer
    les points aberrants, et égale à False si on veut affecter le poid "poid" aux pointx 
    aberrants et un poid = 1 aux points normaux 
    
    entrees :
        x : vecteur
        methode : fonction : x -> a,b
        sup_poid : booleen
        poids : float
        
    sorties :
        x_sup : vecteur
        v_poids : vecteur
        indices : vecteur
    """
    n = len(x)
    a,b = methode(x)
    x_sup = list(x)
    v_poids = [1]*n
    indices = []
    for i,e in enumerate(x):
        if e <a or e>b:
            indices.append(i)
            if sup_poid:
                x_sup[i] = None
            else :
                v_poids[i] = poids
    
    while None in x_sup:
        x_sup.remove(None)
    
    return x_sup,v_poids,indices
            

def supprime_un(x,v_poids,i,methode,sup_poid= 2,poids=1/100):
    """
    la fonction supprime prend un vecteur x, un indice i, le nom une methode de
    detection des points aberrants, un booléen sup_poid égale à 1 si on veut supprimer
    les points aberrants,égale à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
    égale à 3 si on veut remplacer les points aberrants par les valeurs des quantiles
    et égale à 0 si on veut affecter le poid "poid" aux pointx 
    aberrants et un poid = 1 aux points normaux 
    
    entrees :
        x : vecteur
        v_poids : vecteur
        methode : fonction : x -> a,b
        sup_poid : 1,2,3, ou 4
        poids : float
        
    sorties :
        x_sup : vecteur
        v_poids : vecteur
    
    """
    a,b = methode(x)
    xk = x[i]
    x_sup = list(x)
    if xk <a or xk>b:
        if sup_poid == 1:
            x_sup.pop(i)
            v_poids.pop(i)
        elif sup_poid == 2:
            x_sup[i] = None
            v_poids[i] = None
        elif sup_poid == 3:
            x_sup[i] = a if xk<a else b
        else:
            v_poids[i] = poids

    return x_sup,v_poids

if __name__ == "__main__":
    X,Y = ldt.load_points("test.txt")

    
