# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:29:18 2020

@author: amely
"""

# Fonctions utiles
import numpy as np
import scipy.stats as stat
from math import sqrt, floor


####################
# Fonctions utiles #
####################

def moyenne(x, invisibles=None):
    """
    Cette fonction renvoie la moyenne des valeurs de x
    Les éléments dont l'indice est dans la liste d'invisibles sont ignorés
    
    type des entrées :
        x : vecteur de float ou vecteur d'entiers
        invisibles : list[int]
    
    type des sorties :
        float
    """
    n = len(x)
    moyenne = 0.0
    if invisibles == None:
        for val in x:
            moyenne += val
    else:
        moyenne = 0.0
        for i in range(len(x)):
            if i not in invisibles:
                moyenne += x[i]

    moyenne = moyenne / n
    return moyenne


def ecart_type(x, moy, invisibles=None):
    """
    Cette fonction renvoie l'écart type des valeurs de x, à partir de sa moyenne
    Les éléments dont l'indice est dans la liste d'invisibles sont ignorés
    
    type des entrées :
        x : vecteur de float ou vecteur d'entiers
        moy : float
        invisibles : list[int]
    
    type des sorties :
        float
    """
    n = len(x)
    res = 0.0
    if invisibles == None:
        for val in x:
            res += (val - moy) ** 2
    else:
        for i in range(len(x)):
            if i not in invisibles:
                res += (x[i] - moy) ** 2
    res = 1 / n * res
    return sqrt(res)


def calcul_reel(i, indices):
    """
    Calcul l'indice réel de i en sachant que les indices présents dans indices ont été retirés avant, et n'ont donc pas été comptabilisé.
    
    type des entrées :
        i : int
        indices : list[int]
        
    type des sorties :
        int
    """
    indices.sort()  # Pas très optimisé mais pas très gênant car normalement "peu" de points aberrants donc peu d'appels à cette fonction
    i_reel = i
    for k in indices:
        if k <= i_reel:
            i_reel += 1
        else:
            break  # L'indice relatif est avant tous ceux enlevés : ça ne change plus rien. Les autres indices non étudiés dans cette boucle sont encore plus grands.
    return i_reel


def voisinsI(x, i):
    """
    retourne une liste contenant la distance du point à
    la position i aux autres points de x
    :param x: une liste de réels
    :param i: un entier compris entre 0 et la taille de x - 1
    :return: une liste de réels : distance entre x[i] et les autres points
    """
    l = []
    n = len(x)
    for j in range(n):
        if i != j:
            l.append(np.abs(x[i] - x[j]))
    return l




def voisinsKi(x, i, k):
    """
    trie la liste dans l'ordre croissant la liste des distances de i
    à chaque point de x et retourne la moyenne des k plus proches voisins
    c'est à dire sa k-distance
    :param x: une liste de réels
    :param i: un entier i compris entre et la taille de x - 1
    :param k: le nombre de voisins à prendre en compte
    :return: k-distance
    """
    y = sorted(voisinsI(x, i))
    s = 0
    for j in range(k):
        s = s + y[j]
        #try:
            #s = s + y[j]
            #break
        #except IndexError:
             #pass
    return s / k
def KNN_inter(x, k):
    """
    Calculer pour chaque observation la distance au K plus proche voisin k-distance;
    Ordonner les observations selon ces distances k-distance(ordre decroissant)
    :param x: une liste de réels
    :param k: le nombre de voisins à considerer
    :return: une liste de (a,b) a: a element de x et b sa k-distance
    """
    n = len(x)
    l = []
    for i in range(n):
        l.append((x[i], voisinsKi(x, i, k)))
    return sorted(l, key=lambda col: col[1], reverse=True)

def isIN(x,i):
    for j in range(len(x)):
        if i == x[j][0] :
            return True
    return False


#############################################
# Méthodes de détection de points aberrants #
#############################################

def quartile(x, coeff=0.01):
    """
    Méthode inter-quartiles, calcul de l'intervalle.
    La fonction prend une liste de valeurs (ordonnées de points) et renvoie un intervalle [a,b] associé.
    L'intervalle est l'interquartile, étendu des deux côtés du coeff * l'écart.
    Un point sera considéré comme aberrant si son ordonnée n'appartient pas à l'intervalle [a,b]
    
    type des entrées :
        x : vecteur de float ou vecteur d'int
        
    type des sorties :
        (float,float)
        
    """
    x_s = sorted(x)
    n = len(x_s)
    if n < 3:
        return 1, 0  # Intervalle vide : tous les points seront aberrants
    elif n == 3:
        return min(x), max(x)  # Intervalle contenant tous les points, aucun ne sera aberrant
    else:

        k = n // 4
        # le premier quartile
        q1 = x_s[k - 1]
        # le 3éme quartile
        q3 = x_s[3 * k - 1]
        # l'inter-quartile
        inter_q = q3 - q1

    return (q1 - coeff * inter_q, q3 + coeff * inter_q)


def eval_quartile(x, i, a, b):
    """
    Méthode inter-quartiles, test d'aberrance du point.
    Si x[i] appartient à l'intervalle [a,b], renvoie faux, sinon renvoie vrai.
    Renvoie vrai si et seulement si le point n'appartient pas à l'intervalle
    
    type des entrées :
        x : vecteur de float ou vecteur d'int
        i : int
        a : int
        b : int
    
    type des sorties :
        booléen
    """
    return (x[i] < a or x[i] > b)


def test_Chauvenet(x, i):
    """
    Test de Chauvenet
    Renvoie vrai si et seulement si le point x[i] est considéré comme aberrant au regard des autres valeurs de x,
    selon le test de Chauvenet.
    
    type des entrées :
        x : vecteur de float ou vecteur d'int
        i : int
    
    type des sorties :
        booléen
    """
    n = len(x)
    x_barre = moyenne(x)
    var_x = (1 / n) * sum(np.array(x) ** 2) - x_barre ** 2

    # Si la variance est nulle, tous les points sont égaux : aucun d'eux n'est aberrant.
    if var_x == 0:
        return False

    a = abs(x[i] - x_barre) / var_x ** (0.5)
    n_a = (2 * stat.norm.cdf(a, loc=0, scale=1) - 1)
    if n_a > 0.5:
        return True
    else:
        return False


def thompson(x, i, alpha=0.001):
    """
    Test Tau de Thompson
    Renvoie vrai si et seulement si le point x[i] est considéré comme aberrant au regard des autres valeurs de x,
    en considérant une erreur alpha comme acceptable,
    selon le test Tau de Thompson.
    
    type des entrées :
        x : vecteur de float ou vecteur d'int
        i : int
        alpha : float
        
    type des sorties :
        booléen
    """
    n = len(x)
    x_barre = moyenne(x)
    var_x = (1 / n) * sum(np.array(x) ** 2) - x_barre ** 2
    sigma = var_x ** (0.5)
    t_alpha = stat.t.ppf(alpha / 2, n - 1)
    seuil = t_alpha / ((n ** (0.5)) * (n - 2 + t_alpha ** 2) ** (0.5))
    gam = (x[i] - x_barre) / sigma
    if gam > seuil:
        return True
    else:
        return False


def grubbs(x, alpha=5 / 100):
    """
    Test de Grubbs.
    Grubbs est un cas particulier de la déviation extreme de Student.
    La fonction prend une liste de valeurs (ordonnées de points) et un paramètre alpha, le risque d'erreur qu'on accepte.
    L'algorithme de Grubbs est appliqué à la lettre : on applique la formule uniquement sur la valeur la plus éloignée.
    C'est pourquoi il faut appeler cette méthode tant que la valeur renvoyée est vrai mais qu'on n'est pas dans un cas spécial.
    l'indice renvoyé est celui de la valeur extrême, et vaut -1 ou -2 dans les cas spéciaux : écart type nul ou 0 valeurs.
    Elle renvoie une liste de booléens indiquant si la valeur associée est considérée comme aberrante selon le test de Grubbs.
    C'est le cas si la distance à la moyenne empirique est supérieure à un certain seuil.
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        alpha : float
        
    type des sorties :
        booléen, int
    """
    n = len(x)

    if n == 0:
        return False, -2  # False ou True, les deux peuvent être mis ici, aucune coïncidence sur le programme.

    # Calculs de la moyenne et de l'écart type empiriques
    moy = moyenne(x)
    e_t = ecart_type(x, moy)

    if (e_t == 0):  # L'égalité à 0 n'est pas exacte avec les calculs.
        # Les valeurs sont toutes identiques, il n'y a pas de points aberrants
        return False, -1

    # Calculs des distances à la moyennes, normalisées par l'écart type
    dist = [0] * n
    for i in range(n):
        dist[i] = abs(x[i] - moy)

    dist = [d / e_t for d in dist]

    # Calcul de la distance limite
    tcrit = stat.t.ppf(1 - (alpha / (2 * n)),
                       n - 2)  # Valeur critique selon la loi de Student avec n-2 degrés de liberté et une confiance de alpha/2N
    dist_lim = (n - 1) / sqrt(n) * sqrt(tcrit ** 2 / (n - 2 + tcrit ** 2))

    # On cherche la distance maximum avec son indice
    imax = 0
    dmax = 0
    for i in range(n):
        if dist[i] > dmax:
            dmax = dist[i]
            imax = i
    # Si cette distance est plus grande que la limite, la valeur est aberrante.
    return (dmax > dist_lim), imax


# Le test de Tietjen Moore est une généralisation du test de Grubbs.
# Il peut être appliqué peu importe le nombre de valeurs aberrantes
# Mais il faut connaître ce nombre exactement : on n'implémente donc pas cette méthode.

def deviation_extreme_student(x, alpha=5 / 100, borne_max=0):
    """
    En anglais : extreme Studentized deviate (ESD)
    C'est la généralisation du test de Grubbs, sans avoir besoin d'itérer.
    D'après des études de Rosner (Rosner, Bernard (May 1983), Percentage Points for a Generalized ESD Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.) 
    , ce test est très précis pour n >= 25 et reste correct pour n>=15.
    Il faut donc faire attention aux résultats obtenus si on l'appelle sur un intervalle avec peu de points !
    Ce test permet de détecter un ou plusieurs points aberrants, c'est en quelques sortes une généralisation de Grubbs.
    Il nécessite simplement une borne maximale de points aberrants. (qui peut être donnée arbitrairement, par exemple 10% du nombre de points total)
    L'algorithme est appliqué sur les données x. Si la borne maximale vaut 0, alors on considère que c'est 10% du nombre de données (arrondi au supérieur)
    Cette fonction renvoie une liste de booléens indiquant si la valeur associée est considérée comme aberrante.
    Alpha est le risque d'erreur que l'on accepte.
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        alpha : float
        borne_max : int > 0
        
    type des sorties :
        vecteur de booléens de la longueur de x
    """

    ind_candidats = []
    dist_candidats = []
    n = len(x)

    if borne_max == 0:
        borne_max = floor(len(x) / 10) + 1
        if len(x) % 10 == 0:
            borne_max -= 1

    while borne_max != 0:
        moy = moyenne(x, ind_candidats)
        e_t = ecart_type(x, moy, ind_candidats)
        if (e_t == 0):
            break  # Tous les points sont égaux, on ne trouvera pas de points aberrants dans ceux qui restent

        # On calcule la distance des points de la même manière que pour Grubbs, sauf qu'on ne récupère que la distance maximale
        dmax = 0
        ind = 0
        for i in range(n):
            if i not in ind_candidats:
                dtemp = abs(x[i] - moy)
                if dtemp > dmax:
                    ind = i
                    dmax = dtemp
        ind_candidats.append(ind)
        dist_candidats.append(dmax / e_t)

        borne_max -= 1

    i = 0
    # le i des formules devient i-1 car on est ici indicés en 0
    while i != len(ind_candidats):
        # Calculs à partir de Ri
        p = 1 - alpha / (2 * (n - i))
        tcrit = stat.t.ppf(p, n - i - 2)
        seuil = (n - i - 1) * tcrit / sqrt((n - i) * (n - i - 2 + tcrit ** 2))
        if dist_candidats[i] <= seuil:
            break;
        i += 1
    # i-1 est l'indice du dernier point considéré comme aberrant par ce test.

    aberrant = [False] * n
    for j in range(i):
        aberrant[ind_candidats[j]] = True
    return aberrant


#######################################
## METHODE DE K PLUS PROCHES VOISINS ##
#######################################
def KNN(x, j, k, m):
    """
    Calculer pour chaque observation la distance au K plus proche voisin k-distance;
    Ordonner les observations selon ces distances k-distance;
    Les Outliers ont les plus grandes distances k-distance;
    Les observations qui ont les n% plus grandes distances k-distance sont des outliers, n étant un paramètre à fixer
    :param x: une liste de réels
    :param j: l'indice (entier) du point à étudier
    :param k: le nombre de voisins à considerer
    :param m: un entier entre 0 et 100 : représentant le pourcentage de valeurs à rejeter
    :return: un booleen vrai si x[j] est aberrant , faux sinon
    """
    l = []
    z = KNN_inter(x, k)
    p = int((m / 100) * len(z))
    for j in range(p):
        l.append(z[j])
    if isIN(l,x[j]):
        return True
    else:
        return False
