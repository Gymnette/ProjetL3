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

#############################################
# Méthodes de détection de points aberrants #
#############################################

def quartile(x,i,Q=0.01):
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
        return True
    elif n == 3 :
        return False
    else:
        
        k = n//4 
        """
        if (n/4)-(n//4) != 0 :
            k =(n//4) +1
        else:
            k =n//4 +1
        """
        # le premier quartile
        q1 = x_s[k-1]
        # le 3éme quartile
        q3 = x_s[3*k-1]
        # l'inter-quartile
        inter_q = q3-q1
    
        if q1-Q*inter_q < x[i] < q3+Q*inter_q :
            return False
        else:
            return True



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
    #print("x_barre == ",x_barre)
    #print("sigma == ", sigma)
    t_alpha = stat.t.ppf(alpha/2,n-1)
    seuil = t_alpha/((n**(0.5))*(n-2+t_alpha**2)**(0.5))
    gam = (x[i]-x_barre)/sigma 
    #print(i , x[i],  seuil ,  gam)
    if gam > seuil :
        return True
    else :
        return False
        

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

    (x,y) = np.loadtxt('data_CAO.txt')
    
    """
    _________CAS_NON_UNIFORME__________
    """
    
    n =len(x) #même longueur que y
    p = pas_inter(y,epsilon=0.07)
    b = p[0]
    X = []
    Y = []
    M = 1
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
        

    


       
