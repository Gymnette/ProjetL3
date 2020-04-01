# -*- coding: utf-8 -*-
"""
Created on Wed April 1 12:16:02 2020

@author: doumbouya mohamed
"""

import numpy as np

"""
	http://python.jpvweb.com/python/mesrecettespython/doku.php?id=loi_normale 
	lien pour la fonction de calcul de proba de la loi gaussienne utilisée dans
	dans la fonction chauvenet
"""

"""
            METHODE DE CHAUVENET 
            et les fonctions utiles pour son implémentation
"""

def moyenne(x):
    s = 0
    for i in range(len(x)):
        s = s + x[i]
    return s/len(x)

def ecart_type(x):
    m = moyenne(x)
    s = 0
    for i in range(len(x)):
        s = s + (x[i] - m)**2
    return np.sqrt(s/(len(x)-1))

#trouvé sur internet , source dans la documentation plus haut

def pgaussred(x):
    """fonction de répartition de la loi normale centrée réduite
       (= probabilité qu'une variable aléatoire distribuée selon
       cette loi soit inférieure à x)
       formule simplifiée proposée par Abramovitz & Stegun dans le livre
       "Handbook of Mathematical Functions" (erreur < 7.5e-8)
    """
    u = abs(x)  # car la formule n'est valable que pour x>=0

    Z = 1 / (np.sqrt(2 * np.pi)) * np.exp(-u * u / 2)  # ordonnée de la LNCR pour l'absisse u

    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1 / (1 + 0.2316419 * u)
    t2 = t * t
    t4 = t2 * t2

    P = 1 - Z * (b1 * t + b2 * t2 + b3 * t2 * t + b4 * t4 + b5 * t4 * t)

    if x < 0:
        P = 1.0 - P  # traitement des valeurs x<0

    return round(P, 7)  # retourne une valeur arrondie à 7 chiffres


#calcul et renvoie la liste  des valeurs aberrantes
def chauvenet(x):
    moy = np.mean(x) #moyenne
    sd = ecart_type(x) #ecart type
    val_ab = []
    for i in x:
        t = np.abs(i-moy)/sd #t suit loi normale N(0,1)
        na = len(x)*(1-pgcaussred(t)) #n fois P(X>t) = 1 - P(X<=t)
        if na<0.5: #condition de rejet pour la methode de chauvenet
            val_ab.append(i)
    return val_ab #liste contenant les valeurs aberrantes



