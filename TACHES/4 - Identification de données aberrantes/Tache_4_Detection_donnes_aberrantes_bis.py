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

#############################################
# Méthodes de détection de points aberrants #
#############################################

def quartile(x,coeff=0.01):
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

def test_Chauvenet(x,i):
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
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    
    # Si la variance est nulle, tous les points sont égaux : aucun d'eux n'est aberrant.
    if var_x == 0 :
        return False
        
    a = abs(x[i]-x_barre)/var_x**(0.5)
    n_a = (2*stat.norm.cdf(a,loc = 0,scale = 1)-1)
    if n_a > 0.5 :
        return True
    else :
        return False
    
def thompson(x,i,alpha=0.001):
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
    n =len(x)
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    sigma = var_x**(0.5)
    t_alpha = stat.t.ppf(alpha/2,n-1)
    seuil = t_alpha/((n**(0.5))*(n-2+t_alpha**2)**(0.5))
    gam = (x[i]-x_barre)/sigma 
    if gam > seuil :
        return True
    else :
        return False
        

###############################################
# Fonctions de supression de points aberrants #
###############################################

    
def supprime(x,methode,sup_poids= True,poids=1/100):
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
    if M == 3 :
        (a,b) = quartile(y)
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
            if eval_quartile(y,i,a,b) == False :
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
           # c +=1
            p.append(i+1)
        
    # Les deux derniers points appartiendront toujours au dernier intervalle.
    p.append(n)

    return p   


if __name__ == "__main__" :
    ############################
    # Récupération des données #
    ############################
    
    #x,y = ldt.load_points("droite_nulle_pasaberrant.txt")
    #x,y = ldt.load_points("droite_nulle_un_aberrant.txt")
    #x,y = ldt.load_points("droite_environ_nulle_pasaberrant.txt")
    #x,y = ldt.load_points("droite_environ_nulle_aberrant.txt")
    #x,y = ldt.load_points("droite_identite.txt")
    #x,y = ldt.load_points("droite_identite_environ_pasaberrant.txt")
    #x,y = ldt.load_points("droite_identite_environ_aberrant.txt")
    x,y = np.loadtxt('data_CAO.txt')
    
    # signaux de tests (stationnaires uniquement pour l'instant) provenant du générateur
    nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
    
    #x,y, f = stationary_signal((30,), 0.9, noise_func=nfunc)
    #x,y, f = stationary_signal((30,), 0.5, noise_func=nfunc)
    
    # Décommenter ces deux lignes pour faire apparaitre le signal associé
    #xi = np.linspace(0, 1, 100)
    #plt.plot(xi,f(xi))
    
    #######################
    # Choix de la méthode #
    #######################
    
    #M = eval_quartile
    #M = test_Chauvenet
    #M = thompson
    #M = grubbs
    #M = deviation_extreme_student
    
    ##########################
    # Traitement des données #
    ##########################
    
    """
    _________CAS_NON_UNIFORME__________
    """
    
    n =len(x) #même longueur que y
    p = pas_inter(y,epsilon=0.07)
    b = p[0]
    X = []
    Y = []
    M = 3
    i=1
    while i < len(p) : # Tant que i < len(p), il reste une borne droite d'intervalle non utilisée
        a = b
        b = p[i] #On récupère cette borne après avoir décalé
        
        j = x[a:b]
        g = y[a:b]
        
        xd, yd = supp_aberr(j,g,M)
        X = X + xd
        Y = Y + yd
        
        i+=1 # On se décale d'un cran à droite

    if M==1:
        lab = "chauvenet"
    elif M==2 :
        lab ="thompson"
    elif M==3 :
        lab ="quartile"
        
    plt.close('all')
    plt.figure(lab)
    plt.plot(x,y,'b+',label="données aberrantes")
    plt.plot(X,Y,'r+',label="données non aberrantes")
    plt.legend(loc='best')
        

    


       
