from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt

def isIn(l,i):
    for j in l:
        if i==j:
            return True
    return False

def supprime(l,a):
    b = []
    for i in l:
        if not isIn(a,i):
            b.append(i)
    return b


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


def ecart_interquatile(x,weigh=True):
    Q1, Q3 = quantile_13(x)
    ec_interq = Q3 - Q1
    sep_faible = Q1 - 1.5 * ec_interq
    sep_elever = Q3 + 1.5 * ec_interq
    val_ab = []

    for i in range(len(x)):
        if x[i] < sep_faible or x[i] > sep_elever:
            val_ab.append(x[i])
        #else:
            #val_ab.append((x[i], 1))

    return val_ab


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
    return np.sqrt(s/len(x))

#trouvé sur internet , source dans la documentation

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
    sd = ecart_type(x)
    val_ab = []
    for i in x:
        t = np.abs(i-moy)/sd
        na = len(x)*(1-pgaussred(t))
        if na<0.5:
            val_ab.append(i)
    return val_ab

"""
        METHODE DE K PLUS PROCHES VOISINS
        METHODES DES DISTANCES
        
        Calculer pour chaque observation la distance au K plus proche voisin k-distance;
        Ordonner les observations selon ces distances k-distance;
        Les Outliers ont les plus grandes distances k-distance;
        Les observations qui ont les n% plus grandes distances k-distance sont des outliers, n étant un paramètre à fixer.
"""

#retourne une liste contenant la distance de i aux autres points de x
#i est un point de x

def voisinsI(x,i):
    l = []
    for j in x:
        if i!=j:
            l.append(np.abs(i-j))
    return l

#trie la liste dans l'ordre croissant la liste des distances de i
# à chaque point de x et retourne la liste des k premières

def voisinsKi(x,i,k):
    return [np.sort(voisinsI(x,i))[j] for j in range(k)]

#retourne la liste des valeurs aberantes
#en utilisant les deux fonctions spécifiées ci dessus
#les val abe sont n pourcent de
# val de x qui ont les plus grandes k-distance

def KNN(x,k,n=1):
    n = len(x)
    val_ab = []
    d = [] #une liste de couple (a,b) : a val de x et b sa k-distance
    for i in range(n):
        k_distance = np.mean(voisinsKi(x,x[i],k)) #calcul la k-distance de i qui est la moyenne
                                                #de ses k premieres distance
        d.append((x[i],k_distance ))
    d = sorted(d,key=lambda col:col[1],reverse=True)
    p = int(n*len(d)/100) +1
    for j in range(p):
        val_ab.append(d[i])
    return d









if __name__ == '__main__':
    y = [5, 7, 10, 15, 19, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 25]
    #(uk,zk)=np.loadtxt('data.txt')
    val_ab = KNN(y,3)
    print(val_ab)
    """
    d = KNN(y,3)
    print(d)
    s = 0
    e = 0
    for i in range(len(d)):
        s =s + d[i][1]
    s = s/len(d)
    l=[]
    for i in range(len(d)):
        e = e + (d[i][1] - s)**2
    e = np.sqrt(e/len(d))
    for i in range(len(d)):
        if d[i][1] <= s:
            l.append(d[i][0])
    print(e,s)
    #print(l)
    """


