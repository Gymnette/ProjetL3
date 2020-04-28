# -*- coding: utf-8 -*-
# Récupération des tests par fichier ou directement des signaux
import load_tests as ldt
from signaux_splines import *

# Affichage - A MODIFIER AFIN D UTILISER LA LIBRAIRIE D AMELYS
import matplotlib.pyplot as plt

# Fonctions utiles
import numpy as np
import scipy.stats as stat
from math import sqrt,floor


####################
# Fonctions utiles #
####################

def moyenne(x,invisibles=None):
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
    if invisibles == None :
        for val in x :
            moyenne += val
    else :
        moyenne = 0.0
        for i in range(len(x)):
            if i not in invisibles :
                moyenne += x[i]

    moyenne = moyenne / n
    return moyenne


def ecart_type(x,moy,invisibles=None):
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
    if invisibles == None :
        for val in x :
            res += (val-moy)**2
    else :
        for i in range(len(x)):
            if i not in invisibles :
                res += (x[i]-moy)**2
    res = 1/n * res
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
    indices.sort() # Pas très optimisé mais pas très gênant car normalement "peu" de points aberrants donc peu d'appels à cette fonction
    i_reel = i
    for k in indices :
        if k <= i_reel :
            i_reel += 1
        else :
            break #L'indice relatif est avant tous ceux enlevés : ça ne change plus rien. Les autres indices non étudiés dans cette boucle sont encore plus grands.
    return i_reel

#############################################
# Méthodes de détection de points aberrants #
#############################################

def quartile(x,coeff=0.5):
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
    if n <3 :
        return 1,0 # Intervalle vide : tous les points seront aberrants
    elif n == 3 :
        return min(x),max(x) #Intervalle contenant tous les points, aucun ne sera aberrant
    else:

        k = n//4
        # le premier quartile
        q1 = x_s[k-1]
        # le 3éme quartile
        q3 = x_s[3*k-1]
        # l'inter-quartile
        inter_q = q3-q1

    return (q1-coeff*inter_q,q3+coeff*inter_q)

def eval_quartile(x,i,a,b):
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

def test_Chauvenet(x,i,tau=0.5):
    """
    cette fonction prend un vecteur x et un indice i
    et qui revoie True si le pois x[i] est consideré comme un point
    aberrant selon le test de Chauvenet, et renvoie False sinon
    """
    n = len(x)
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    #print("var_x = ", var_x**0.5)
    a = abs(x[i]-x_barre)/var_x**(0.5)
    """
    verification du calcul du proba
    """
    n_a = n*(1-stat.norm.cdf(a,loc = 0,scale = 1))
    #print("n_a = ", n_a)
    if n_a < tau :
        return True
    else :
        return False

def thompson(x,i,alpha=1.995):
    """
    cette fonction prend un vecteur x, un indice i et un parametre alpha
    et qui revoie True si le pois x[i] est un point aberrant, et
    renvoie False sinon
    """
    n =len(x)
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    sigma = var_x**(0.5)
    #print("x_barre == ",x_barre)
    #print("sigma == ", sigma)
    t_alpha = stat.t.ppf(alpha/2,n-1)
    seuil = (t_alpha*(n-1))/((n**(0.5))*(n-2+t_alpha**2)**(0.5))
    gam = (x[i]-x_barre)/sigma
    #print(i , x[i],  seuil ,  gam)
    if gam > seuil or gam < -seuil  :
        return True
    else :
        return False

def grubbs(x,alpha=5/100):
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

    if n == 0 :
        return False, -2 # False ou True, les deux peuvent être mis ici, aucune coïncidence sur le programme.


    # Calculs de la moyenne et de l'écart type empiriques
    moy = moyenne(x)
    e_t = ecart_type(x,moy)

    if (e_t == 0 ): # L'égalité à 0 n'est pas exacte avec les calculs.
        #Les valeurs sont toutes identiques, il n'y a pas de points aberrants
        return False, -1

    # Calculs des distances à la moyennes, normalisées par l'écart type
    dist = [0]*n
    for i in range(n) :
        dist[i] = abs(x[i]-moy)

    dist = [d/e_t for d in dist]

    # Calcul de la distance limite
    tcrit = stat.t.ppf(1-(alpha/(2*n)),n-2)# Valeur critique selon la loi de Student avec n-2 degrés de liberté et une confiance de alpha/2N
    dist_lim = (n-1)/sqrt(n) * sqrt(tcrit**2 / (n-2+tcrit**2))

    # On cherche la distance maximum avec son indice
    imax = 0
    dmax = 0
    for i in range(n):
        if dist[i] > dmax :
            dmax = dist[i]
            imax = i
    # Si cette distance est plus grande que la limite, la valeur est aberrante.
    return (dmax > dist_lim),imax

# Le test de Tietjen Moore est une généralisation du test de Grubbs.
# Il peut être appliqué peu importe le nombre de valeurs aberrantes
# Mais il faut connaître ce nombre exactement : on n'implémente donc pas cette méthode.

def deviation_extreme_student(x,alpha=5/100, borne_max=0):
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

    if borne_max == 0 :
        borne_max = floor(len(x)/10)+1
        if len(x)%10 == 0 :
            borne_max -= 1


    while borne_max != 0 :
        moy = moyenne(x,ind_candidats)
        e_t = ecart_type(x,moy,ind_candidats)
        if (e_t == 0) :
            break # Tous les points sont égaux, on ne trouvera pas de points aberrants dans ceux qui restent

        # On calcule la distance des points de la même manière que pour Grubbs, sauf qu'on ne récupère que la distance maximale
        dmax = 0
        ind = 0
        for i in range(n) :
            if i not in ind_candidats :
                dtemp = abs(x[i]-moy)
                if dtemp > dmax :
                    ind = i
                    dmax = dtemp
        ind_candidats.append(ind)
        dist_candidats.append(dmax/e_t)

        borne_max -= 1

    i = 0
    # le i des formules devient i-1 car on est ici indicés en 0
    while i != len(ind_candidats):
        # Calculs à partir de Ri
        p = 1 - alpha/(2*(n-i))
        tcrit = stat.t.ppf(p,n-i-2)
        seuil = (n-i-1)*tcrit / sqrt((n-i)*(n-i-2+tcrit**2))
        if dist_candidats[i] <= seuil :
            break;
        i += 1
    # i-1 est l'indice du dernier point considéré comme aberrant par ce test.

    aberrant = [False]*n
    for j in range(i):
        aberrant[ind_candidats[j]] = True
    return aberrant


def voisinsI(x, y, a, b,k):
    """
    :param x: une liste de réels d'abscisses
    :param y: une liste de réels d'ordonnées
    :param a: un réel abscisse
    :param b: un réel ordonné
    :return: la distance de point (a,b) au plus proche point dans x,y
    """
    n = len(y)
    l = []
    for j in range(n):
        if y[j] != b and a != x[j]:
            l.append(np.sqrt((b - y[j])**2 + (a - x[j])**2))
    l = sorted(l)
    s = 0
    for i in range(k):
        s += l[i]
    return s/k


# retourne la liste des valeurs aberantes
# en utilisant les deux fonctions spécifiées ci dessus
# les val abe sont n pourcent de
# val de x qui ont les plus grandes k-distance

def KNN(x, y, k, m):
    """
    :param x: une liste de réels (abscisses)
    :param y: une liste de réels (ordonnées)
    :param k: entier, le nombre de voisins à prendre
    :param m: un entier, pourcentage de valeurs à rejetter
    :return: 4 listes : les 2 1ères representent les abscisses
    et ordonnées de valeurs aberrantes et les 2 dernières celles
    non aberrantes
    """
    n = len(y)
    if k >= n:
        print("le nombre de voisins à prendre en compte est supérieure à la taille des données")
        exit(1)
    x_ab = []
    y_ab = []
    x_nab = []
    y_na = []
    l = list()
    for i in range(n):
        l.append(( x[i],y[i], voisinsI(x, y, x[i],y[i],k) ))
    z = sorted(l, key=lambda col: col[2], reverse=True)
    p = int((m / 100) * len(z))
    for i in range(p):
        x_ab.append(z[i][0])
        y_ab.append(z[i][1])
    for j in range(p, len(z)):
        x_nab.append(z[j][0])
        y_na.append(z[j][1])
    return x_nab, y_na




###############################################
# Fonctions de supression de points aberrants #
###############################################


def supprime(x,methode,sup_poids= True,poids=1/100): #A AJOUTER (AMELYS) : OPTIONS DES METHODES
    """
    Parcours toutes les valeurs de x afin de toutes les traiter.
    La fonction supprime prend un vecteur x d'ordonnées de points, une methode de
    detection des points aberrants, un booléen sup_poids égal à True si on veut supprimer
    les points aberrants, et égal à False si on veut affecter le poids "poids" aux points
    aberrants et un poids = 1 aux points considérés comme adaptés.
    Elle renvoie une liste d'ordonnées ne contenant pas celles supprimées,
    une liste de poids, ainsi qu'une liste des indices dans le vecteur des valeurs supprimées.

    type des entrees :
        x : vecteur de float ou vecteur d'int
        methode : fonction :vecteur de float ou vecteur d'int -> (float,float)
        sup_poids : booleen
        poids : float

    type des sorties : tuple (x_sup,v_poids,indices)
        x_sup : vecteur de float ou vecteur d'int
        v_poids : vecteur de float
        indices : vecteur d'int
    """
    n = len(x)
    x_sup = list(x)
    v_poids = [1]*n
    indices = []

    if methode == eval_quartile :
        a,b = quartile(x)

    if methode == grubbs :
        res, ind = grubbs(x)
        x_cpy = list(x)
        while (ind >=0 and res ) : #Un point aberrant a été trouvé de manière "classique".
            ind_reel = calcul_reel(ind,indices)
            indices.append(ind_reel)
            if sup_poids:
                x_sup[ind_reel] = None
            else :
                v_poids[ind_reel] = poids

            x_cpy.pop(ind) # c'est bien ici le relatif
            res, ind = grubbs(x_cpy)
        # Si c'est res qui est faux, pas de soucis, on a notre résultat.
        # Si l'indice est négatif, le résultat sera faux, donc c'est bon, pas de point aberrant détecté.
    elif methode == deviation_extreme_student :
        est_aberrant = methode(x)
        for i in range(n):
            if est_aberrant[i] :
                indices.append(i)
                if sup_poids:
                    x_sup[i] = None
                else :
                    v_poids[i] = poids
    else :

        for i in range(n):
            aberrant = False
            if methode == test_Chauvenet or methode == thompson:
                if methode(x,i):
                    aberrant = True
            else : #methode == eval_quartile:
                if eval_quartile(x,i,a,b):
                    aberrant = True

            if aberrant :
                indices.append(i)
                if sup_poids:
                    x_sup[i] = None
                else :
                    v_poids[i] = poids

    while None in x_sup:
        x_sup.remove(None)

    return x_sup,v_poids,indices



def supprime_un(x,v_poids,i,methode,sup_poids= 2,poids=1/100): #COMMENTAIRE BERYL : PAS TOUCHEE, JE TE LE LAISSE AMELYS
    """
    Traite une valeur de x, donnée par l'indice i.
    La fonction supprime prend un vecteur x d'ordonnées, le vecteur des poids associés,
    un indice i de l'élément à supprimer, une méthode de détection des points aberrants, u
    n entier sup_poids égal à 1 si on veut supprimer les points aberrants,
    égal à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
    égal à 3 si on veut remplacer les points aberrants par les valeurs non aberrantes les plus proches (Méthode de Winsorising) :
        - Affecte le quartile le plus proche pour la méthode interquartile
        DECRIRE ICI CE QUI EST FAIT POUR LES AUTRES METHOOOOOOOOOOOOOOOODES
    et égal à 0 si on veut affecter le poids "poids" aux points aberrants et un poids = 1 aux points normaux.

    type des entrees :
        x : vecteur de float ou vecteur d'int
        v_poids : vecteur de float
        methode : fonction : vecteur de float ou vecteur d'int -> (float, float)
        sup_poid : 0,1,2,3
        poids : float

    type des sorties : couple (x_sup, v_poids)
        x_sup : vecteur de float ou vecteur d'int
        v_poids : vecteur de float

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


def supp_aberr(x,y,M=1) :
    """
    cette foction supprime les points (xi,yi) s'ils sont considéré comment
    des points aberrants
    le parametre M prend trois valeurs {1,2,3}, 1 si on veut utiliser
    la méthode de Chauvenet, 2 la méthode de thompson, 3 la méthode
    d'inter-quartile
    """
    x_d = []
    y_d = []

    for i in range(len(x)):
        if M==1:
            if test_Chauvenet(y,i) == False :
                x_d.append(x[i])
                y_d.append(y[i])

        elif M==2 :
            if thompson(y,i) == False :
                x_d.append(x[i])
                y_d.append(y[i])

        elif M==3:
            if quartile(y,i) == False :
                x_d.append(x[i])
                y_d.append(y[i])

    return x_d, y_d

###################################
# Gestion des intervalles d'étude #
###################################

def pas_inter(y,epsilon=0.1):
    """
    Cette fonction prend un vecteur y et un paramètre de variation epsilon,
    et renvoie des intervalles sur lesquels la variation de y est inferieure à epsilon.
    Les intervalles sont représentés par une liste d'entiers, dont l'ordre est important :
    chaque entier représente un intervalle en indiquant le début de celui ci, excepté la dernière valeur indiquant la fin du dernier, exclue.
    L'intervalle représenté à l'indice i est donc [p[i],p[i+1][

    Type des entrées :
        y : vecteur de float ou vecteur d'int
        epsilon : float

    Type des sorties :
        liste[int]
    """
    p = [0]
    n = len(y)
    for i in range(n-2):
        d_yi = y[i+1]-y[i]
        d_yi_1 = y[i+2]-y[i+1]
        delta = abs(d_yi - d_yi_1)
        if delta > epsilon :
            p.append(i+1)

    # Les deux derniers points appartiendront toujours au dernier intervalle.
    p.append(n)

    return p

def esti_epsilon(y):
        n = len(y)
        d_yi = y[1:n]-y[0:n-1]
        delta = abs(d_yi[1:n-1] - d_yi[0:n-2])

        for i in range(len(delta)):
            if test_Chauvenet(delta,i) == False :
                list(delta).pop(i)


        return sum(delta)/len(delta)



def densite(x,d,f):
    """
    cette fonction prend un vecteur x et un intervalle [d,f],et renvoie la densité
    des points sur l'intervalle[x[d],x[f]]
    """
    j = x[d:(f+1)]
    return len(j)/abs(x[f]-x[d])




def ind_int(x,d):
    """
    cette fonction prend un vecteur x et un entier d , et elle renvoie un entier i,
    tel que la densité des points sur l'intervalle [x[d],x[i]] est maximale
    """
    n = len(x)
    i =d+1
    while(i < n-1 ):
        ds1 = densite(x,d,i)
        ds2 = densite(x,d,i+1)
        if ds2 < ds1 :
            return i
        else:
            i+=1
    return n-1

def ind_densite(x):
    """
    cette fonction prend un vecteur x et elle renvoie une liste des indices
    des intervalles les plus denses
    """
    p = [0]
    n = len(x)
    i =0
    while i < n-2 :
        i = ind_int(x,i)
        p.append(i)
        if i == n-1 :
            break

    return p

def regrouper(p,t=8):
    """
    cette fonction regroupe les intervalles de taille inferieure à t avec leurs voisins
    """
    i = 0
    n = len(p)
    while i < n-2:
        if (p[i+1]-p[i]) < t :
            p.pop(i+1)
        else:
            i+=1
        n = len(p)
    return p


def pas_inter_essai(y,epsilon=0.1):
    """
    Cette fonction prend un vecteur y et un paramètre de variation epsilon,
    et renvoie des intervalles sur lesquels la variation de y est inferieure à epsilon.
    Les intervalles sont représentés par une liste d'entiers, dont l'ordre est important :
    chaque entier représente un intervalle en indiquant le début de celui ci, excepté la dernière valeur indiquant la fin du dernier, exclue.
    L'intervalle représenté à l'indice i est donc [p[i],p[i+1][

    Type des entrées :
        y : vecteur de float ou vecteur d'int
        epsilon : float

    Type des sorties :
        liste[int]
    """
    p = [0]
    n = len(y)
    for i in range(n-2):
        d_yi = abs(y[i+1]-y[i])
        d_yi_1 = abs(y[i+2]-y[i])
        print(d_yi,d_yi_1,i)

        if (d_yi > epsilon and d_yi_1 > epsilon):
            print(i)
            p.append(i+1)
        if d_yi > epsilon and d_yi_1 <= epsilon :
            i += 1 # Il y a eu un point "aberrant"(c'est bête de ne pas le retirer tout de suite...)


    # Les deux derniers points appartiendront toujours au dernier intervalle.
    p.append(n)

    return p


if __name__ == "__main__" :
    ############################
    # Récupération des données #
    ############################

    # POUR ZAKARIA : ATTENTION, LA PLUPART DE CES TESTS NE "FONCTIONNENT PAS", A REGARDER
    #x,y = ldt.load_points("droite_nulle_pasaberrant.txt")
    #x,y = ldt.load_points("droite_nulle_un_aberrant.txt")
    #x,y = ldt.load_points("droite_environ_nulle_pasaberrant.txt")
    #x,y = ldt.load_points("droite_environ_nulle_aberrant.txt")
    #x,y = ldt.load_points("droite_identite.txt")
    #x,y = ldt.load_points("droite_identite_environ_pasaberrant.txt")
    #x,y = ldt.load_points("droite_identite_environ_aberrant.txt")
    #x,y = np.loadtxt('data_CAO.txt')

    #seed = 15833262
    #seed = 73069674
    #seed = 5505361
    #seed = 50204107
    #46337323
    #3062448
    #34103609
    #75898075
    #seed=48826821
    #21561409

    # signaux de tests (stationnaires uniquement pour l'instant) provenant du générateur
    nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)

    # Seed sert à "fixer" le test
    x,y, f, seed = stationary_signal((200,), 0.9, noise_func=nfunc,seed=5505361)
    #x,y, f, seed = stationary_signal((300,), 0.5, noise_func=nfunc)

    #seed = 94792868
    #seed=32427415
    #95109780
    #40519673
    #91606489
    #87608280
    #13274649
    #44241148

    # signal non stationnaire bruité de 30 points
    #x, y, f, seed = non_stationary_signal((300,), switch_prob=0.1, noise_func=nfunc)

    #######################
    # Choix de la méthode #
    #######################

    #M = eval_quartile
    #M = test_Chauvenet
    M = thompson
    #M = grubbs
    #M = deviation_extreme_student

    #############################################################
    # Epsilon à choisir en fonction des graines et des méthodes #
    #############################################################
    # Pour les signaux stationnaires de paramètres 30, et 0.9
    # Pour les paramètres des méthodes par défaut
    #           0      1       2       3       4        5
    #Quartile  0.5
    #Chauvenet
    #Thompson
    #Grubbs    0.3
    #ESD       0.3

    ##########################
    # Traitement des données #
    ##########################


    #x = list(x)
    n =len(x) #même longueur que y
    ep = esti_epsilon(y)
    #p = pas_inter(y,epsilon = ep) #ESSAI
    p = ind_densite(y)

    if M == eval_quartile :
        p = regrouper(p,t=10)

    b = p[0]
    X = []
    Y = []
    i=1
    while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
        a = b
        b = p[i] #On récupère cette borne après avoir décalé

        j = x[a:b+1]
        g = y[a:b+1]


        yd,v_poids,indices_aberrants = supprime(g,M) #AMELYS : IL FAUT GERER LE CAS Où ON NE SUPPRIME PAS LES POIDS
        indices_aberrants.sort()
        # On parcourt les indices dans l'ordre décroissant pour ne pas avoir de décalage
        # On ne garde que les x associés aux y.
        xd = list(j)
        for ind in range(len(indices_aberrants)-1,-1,-1): #On part de la fin pour ne pas avoir de décalage d'indices
            xd.pop(indices_aberrants[ind])

        X = X + xd
        Y = Y + yd

        i+=1 # On se décale d'un cran à droite

    if M == eval_quartile:
        lab = "Méthode interquartile"
    elif M == test_Chauvenet:
        lab = "Test de Chauvenet"
    elif M == thompson:
        lab = "Méthode Tau de Thompson"
    elif M == grubbs :
        lab = "Test de Grubbs"
    elif M == deviation_extreme_student:
        lab = "Test de la déviation extreme de student"
    else :
        print("Méthode inconnue")
        exit(1)

    plt.close('all')
    plt.figure(lab)
    plt.plot(x,y,'b+',label="données")
    plt.plot(X,Y,'or',color='r',label="données conservées, dites \" non aberrantes\" ")
    plt.legend(loc='best')
    # Décommenter ces deux lignes pour faire apparaitre le signal associé
    x = np.linspace(0, 1, 100)
    plt.plot(x,f(x))




    """
        LE TEST DE LA METHODE DE MOHAMED
    """


    """
    n =len(x) #même longueur que y
    ep = esti_epsilon(y)
    p = pas_inter(y,epsilon = ep) #ESSAI
    #p = ind_densite(y)
    p = regrouper(p,30)

    b = p[0]
    X = []
    Y = []
    i=1
    while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
        a = b
        b = p[i] #On récupère cette borne après avoir décalé

        j = x[a:b+1]
        g = y[a:b+1]
        k = (b-a+1)//2

        xd, yd = KNN(j,g,k,15) #AMELYS : IL FAUT GERER LE CAS Où ON NE SUPPRIME PAS LES POIDS



        X = X + xd
        Y = Y + yd

        i+=1 # On se décale d'un cran à droite


    plt.figure('KMN')
    plt.plot(x,y,'b+',label="données")
    plt.plot(X,Y,'or',color='r',label="données conservées, dites \" non aberrantes\" ")
    plt.plot(x, f(x), 'b', label='signal')
    plt.legend(loc='best')
    # Décommenter ces deux lignes pour faire apparaitre le signal associé
    #xi = np.linspace(0, 1, 100)
    #plt.plot(xi,f(xi))
    """

