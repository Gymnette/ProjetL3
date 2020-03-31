# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:43:04 2020

@author: zakaria
"""


import numpy as np 
import matplotlib.pyplot as plt

"""
la fonction prend un vecteur et qui renvoie un intervalle [a,b]
un point est considéré aberrant s'il n'appartient pas à intervalle [a,b]
"""
def quartile(x):
    n = len(x)
    k = n/4
    if (n/4)-int(n/4) != 0 :
        k =int(n/4) +1
    else:
        k = n//4
    # le premier quartile
    q1 = x[k-1]
    # le 3éme quartile
    q3 = x[3*k-1]
    # l'inter-quartile
    inter_q = q3-q1
    
    return q1-1.5*inter_q, q3+1.5*inter_q


            
"""
la fonction supprime prend un vecteur x, un indice i, le nom une methode de
detection des points aberrants, un booléen sup_poid égale à True si on veut supprimer
les points aberrants, et égale à False si on veut affecter le poid "poid" aux pointx 
aberrants et un poid = 1 aux points normaux 
"""
def supprime(x,methode,sup_poid= True,poids=1/100):
    n = len(x)
    a,b = methode(x)
    x_sup = list(x)
    v_poids = [1]*n
    
    for i,e in enumerate(x):
        if e <a or e>b:
            if sup_poid:
                x_sup[i] = None
            else :
                v_poids[i] = poids
    
    while None in x_sup:
        x_sup.remove(None)
    
    return x_sup,v_poids
            
"""
la fonction supprime prend un vecteur x, un indice i, le nom une methode de
detection des points aberrants, un booléen sup_poid égale à 1 si on veut supprimer
les points aberrants,égale à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
égale à 3 si on veut remplacer les points aberrants par les valeurs des quantiles
et égale à 0 si on veut affecter le poid "poid" aux pointx 
aberrants et un poid = 1 aux points normaux 
"""
def supprime_un(x,v_poids,i,methode,sup_poid= 2,poids=1/100):
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

    x = [5,7,10,15,19,21,21,22,23,23,23,23,23,24,24,24,24,25]
    a,b = quartile(x)

    xb,vb = supprime(x,quartile,False)
    
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