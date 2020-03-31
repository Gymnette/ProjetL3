# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:43:04 2020

@author: zakaria
"""


import numpy as np 
import matplotlib.pyplot as plt


def quartile(x):
    """
    la fonction prend un vecteur et qui renvoie un intervalle [a,b]
    un point est considéré aberrant s'il n'appartient pas à intervalle [a,b]
    """

    n = len(x)
    k = n/4
    if (n/4)-int(n/4) != 0 :
        k =int(n/4) +1
    # le premier quartile
    q1 = x[k-1]
    # le 3éme quartile
    q3 = x[3*k-1]
    # l'inter-quartile
    inter_q = q3-q1
    
    return q1-1.5*inter_q, q3+1.5*inter_q



    
def supprime_un(x,x_sup,v_poids,i,methode,sup_poid= 2,poids=1/100):
    a,b = methode(x)
    #x_sup = list(x)
    xk = x_sup[i]
    
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




if __name__ == "__main__" :

    x = [5,7,10,15,19,21,21,22,23,23,23,23,23,24,24,24,24,25]
    print(x)
    v_poids = [1]*len(x)
    #for i in range(len(x)):
    
    i=0
    x_sup = x
    
    while i < len(x_sup) :
        n = len(x_sup)
        x_sup, v_poids = supprime_un(x,x_sup,v_poids,i,quartile,1)
        if len(x_sup) == n:
            i+=1

    print("____________")
    print(x_sup)
    print(v_poids)
    
    
    
    
    
    
    
    