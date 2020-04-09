# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:09:26 2020

@author: amelys
"""

import numpy as np

def load_points(fichier):
    (X,Y) = np.loadtxt(fichier)
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