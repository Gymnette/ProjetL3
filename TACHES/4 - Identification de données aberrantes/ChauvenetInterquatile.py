import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.crackfortran import n

"""         1ere METHODE DE DETECTION DE DONNEES ABERRANTES
            Identification des valeurs aberrantes avec la règle 1,5 x écart interquartile 
            une valeur aberrante est dite faible si elle est inférieure à
            Q1 -1,5*EcartInterquatile et élevée si elle est supérieure à 
            Q3 +1,5*EcartInterquatile , EcartInterquatile = Q3 - Q1
"""


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


def ecart_interquatile(x,weigh):
    Q1, Q3 = quantile_13(x)
    ec_interq = Q3 - Q1
    sep_faible = Q1 - 1.5 * ec_interq
    sep_elever = Q3 + 1.5 * ec_interq
    val_ab = []

    for i in range(len(x)):
        if x[i] < sep_faible or x[i] > sep_elever:
            val_ab.append((x[i],weigh))
        else:
            val_ab.append((x[i], 1))

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
    return np.sqrt(s/(len(x)-1))

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



if __name__ == '__main__':
    #x = [5, 7, 10, 15, 19, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 25]
    x  = [35.9, 36, 36, 36.2, 36.1, 35.2]
    val_ab = chauvenet(x)
    val_abs = ecart_interquatile(x,0.1)
    print(val_ab)
    print(val_abs)
