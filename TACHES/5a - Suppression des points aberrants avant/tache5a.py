# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt


def quartile(x, param=1.5):
    """
    Cette fonction prend un vecteur et renvoie un intervalle [a,b]
    un point est considéré comme aberrant s'il n'appartient pas à l'intervalle [a,b]
    """
    n = len(x)
    k = n//4 if n%4 == 0 else n//4+1
    # le premier quartile
    q1 = x[k-1]
    # le 3éme quartile
    q3 = x[3*k-1]
    # l'inter-quartile
    inter_q = q3-q1
    
    return q1-param*inter_q, q3+param*inter_q


            

def supprime(x,methode,sup_poids= True,poids=1/100):
    """
    Cette fonction supprime prend un vecteur x, une methode de
    detection des points aberrants, un booléen sup_poids égal à True si on veut supprimer
    les points aberrants, et égal à False si on veut affecter le poids "poids" aux points
    aberrants et un poids = 1 aux points normaux 
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
    la fonction supprime prend un vecteur x, un indice i, le nom une methode de
    detection des points aberrants, un booléen sup_poids égale à 1 si on veut supprimer
    les points aberrants,égale à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
    égale à 3 si on veut remplacer les points aberrants par les valeurs des quantiles
    et égale à 0 si on veut affecter le poid "poids" aux pointx 
    aberrants et un poids = 1 aux points normaux 
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

    x = [5,7,10,15,19,21,21,22,23,23,23,23,23,24,24,24,24,25]
    T = [i for i in range(len(x))]
    
    plt.scatter(T,x,c = 'r',label = "points donnes",s = 100,marker = 'x')

    xb,vb,indb = supprime(x,quartile,True)
    Tb = list(T)
    indb.reverse()
    for ind in indb:
        Tb.pop(ind)
        
    plt.scatter(Tb,xb,label = "points non aberrants",s = 50)
    plt.legend()
    vc = [1]*len(x)
    xc = list(x)
    for i in range(len(x)):
        xc,vc = supprime_un(xc,vc,i,quartile,sup_poid=3)
        
    vd = [1]*len(x)
    xd = list(x)
    i = 0
    n = len(xd)
    while i < n:
        xd,vd = supprime_un(xd,vd,i,quartile,sup_poid=1)
        if len(xd)==n:
            #on augmente i seulement si on a rien supprime, sinon on sauterai une valeur
            i+=1
        n = len(xd)
    
    print("bloup :",xb,vb,sep = '\n')
    print("bloup2 :",xc,vc,sep = '\n')
    print("bloup3 :",xd,vd,sep = '\n')
    
