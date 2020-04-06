# -*- coding: utf-8 -*-
# Récupération des tests par fichier ou directement des signaux
import load_tests as ldt
from signaux_splines import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat


# calcul le 1er et le 3e quartile de x
# triee dans l'ordre croissant
# peut être qu'il existe une fonction numpy capable de
# calculer les quartiles qui peut eventuellement remplacer cette fonction

def quantile_13(x):
    n = len(x)
    if n % 4 == 1:
        Q1 = x[n // 4 - 1]
        Q3 = x[3 * n // 4 - 1]
    else:
        Q1 = x[n // 4]
        Q3 = x[3 * n // 4]
    return Q1, Q3


"""
    fonction detectant les valeurs aberrantes dans x selon la methode
    decrite plus haut et affecte le poids weigh aux pts aberrants, x triee dans l'ordre croissant
    retourne un liste de couples (a,b): dont a est un point de x et b son poids(1 ou weigh)
    si b = weigh , a est un point aberrant
"""


def ecart_interquatile(x, i):
    Q1, Q3 = quantile_13(x)
    ec_interq = Q3 - Q1
    sep_faible = Q1 - 1.5 * ec_interq
    sep_elever = Q3 + 1.5 * ec_interq
    val_ab = []
    for i in range(len(x)):
        if x[i] < sep_faible or x[i] > sep_elever:
            val_ab.append(x[i])
    return val_ab


"""
            METHODE DE CHAUVENET 
            et les fonctions utiles pour son implémentation
"""

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
    moy = np.mean(x)
    sd = np.std(x)
    val_ab = []
    val_na = []
    for i in x:
        t = np.abs(i-moy)/sd
        na = len(x)*(1-pgaussred(t))
        if na<0.5:
            val_ab.append(i)
        else:
            val_na.append(i)
    return val_ab,val_na



"""
        METHODE DE K PLUS PROCHES VOISINS
        METHODES DES DISTANCES

        Calculer pour chaque observation la distance au K plus proche voisin k-distance;
        Ordonner les observations selon ces distances k-distance;
        Les Outliers ont les plus grandes distances k-distance;
        Les observations qui ont les n% plus grandes distances k-distance sont des outliers, n étant un paramètre à fixer.
"""


# retourne une liste contenant la distance du point à la position i aux autres points de x


def voisinsI(x, i):
    l = []
    n = len(x)
    for j in range(n):
        if i != j:
            l.append(np.abs(x[i] - x[j]))
    return l


# trie la liste dans l'ordre croissant la liste des distances de i
# à chaque point de x et retourne la moyenne des k plus proches voisins
# c'est à dire sa k-distance

def voisinsKi(x, i, k):
    y = sorted(voisinsI(x, i))
    s = 0
    for j in range(k):
        s = s + y[j]
    return s / k



# retourne la liste des valeurs aberantes
# en utilisant les deux fonctions spécifiées ci dessus
# les val abe sont n pourcent de
# val de x qui ont les plus grandes k-distance

def KNN(x, y, k, m):
    n = len(y)
    x_ab = []
    y_ab = []
    x_nab = []
    y_na = []
    l = []
    for i in range(n):
        l.append((x[i],y[i], voisinsKi(y, i, k)))
    z = sorted(l, key=lambda col: col[2], reverse=True)
    p = int((m / 100) * len(z))
    for i in range(p):
        x_ab.append(z[i][0])
        y_ab.append(z[i][1])
    for j in range(p, len(z)):
        x_nab.append(z[j][0])
        y_na.append(z[j][1])
    return x_ab,y_ab, x_nab,y_na



if __name__ == "__main__":
    (uk, uz) = np.loadtxt('data.txt')
    uk_xa,uk_ya,uz_xna,uz_yna = KNN(uk,uz,3,28)
    plt.figure("k plus proches voisins")
    plt.plot(uk_xa,uk_ya,'rx',color='b',label='données aberrantes')
    plt.plot(uz_xna, uz_yna, 'or', color='y', label='données non aberrantes')
    plt.legend(loc=True)
    plt.show()
