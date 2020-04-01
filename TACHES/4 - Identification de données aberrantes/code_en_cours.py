# -*- coding: utf-8 -*-
import load_tests as ldt
import scipy.stats as stat
from math import sqrt
import matplotlib.pyplot as plt

#############################################
# Méthodes de détection de points aberrants #
#############################################

# Problème de local/global
# Pour avoir des résultats fiables (dans le cas contraire, on risque de ne pas détecter certaines valeurs aberrantes)
# Plus les valeurs sont denses, plus il faudrait en considérer localement.
    
def quartile(x):
    """
    Méthode inter-quartiles.
    La fonction prend une liste de valeurs (ordonnées de points) et renvoie un intervalle [a,b] associé.
    Un point sera considéré comme aberrant si son ordonnée n'appartient pas à l'intervalle [a,b]
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        
    type des sorties :
        (float,float)
        
    """
    n = len(x)
    k = n//4 if n%4 == 0 else n//4+1
    # le premier quartile
    q1 = x[k-1]
    # le 3éme quartile
    q3 = x[3*k-1]
    # l'inter-quartile
    inter_q = q3-q1
    
    return q1-1.5*inter_q, q3+1.5*inter_q

def Grubbs(x,alpha=5/100):
    # Ce test n'est pas recommandé dans le cas où il y a plus d'une valeur aberrante.
    # Par intervalle étudié, il n'y en a que très peu, on peu supposer qu'il n'y en a qu'une.
    # On est donc dans un cas d'application de ce test.
    """
    Test de Grubbs.
    La fonction prend une liste de valeurs (ordonnées de points) et un paramètre alpha, le risque d'erreur qu'on accepte.
    Elle renvoie une liste de booléens indiquant si la valeur associée est considérée comme aberrante selon le test de Grubbs.
    C'est le cas si la distance à la moyenne empirique est supérieure à un certain seuil.
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        alpha : float
        
    type des sorties :
        vecteur de boolean de la longueur de x
    """
    n = len(x)
    
    # Calculs de la moyenne et de l'écart type empiriques
    moyenne = 0.0
    for val in x :
        moyenne += val
    moyenne = moyenne / n
    
    # Calculs des "distances" en parallèle de l'écart type
    dist = [0]*n
    ecart_type = 0.0
    for i in range(n) :
        val = x[i]
        ecart_type += (val-moyenne)**2
        dist[i] = abs(val-moyenne)
    ecart_type = 1/n * ecart_type
    ecart_type = sqrt(ecart_type)
    
    if (ecart_type == 0): #Les valeurs sont toutes identiques
        return [False]*n
    
    dist = [d/ecart_type for d in dist]
    
    # Calcul de la distance limite
    tcrit = stat.t.ppf(1-(alpha/(2*n)),n-2)# Valeur critique selon la loi de Student avec n-2 degrés de liberté et une confiance de alpha/2N    
    dist_lim = (n-1)/sqrt(n) * sqrt(tcrit**2 / (n-2+tcrit**2))
    
    aberrant = []
    for i in range(n):
        aberrant.append((dist[i] > dist_lim))
    return aberrant
    
    
def deviation_extreme_student():
    """
    En anglais : extreme Studentized deviate (ESD)
    """
    
def Tietjen_Moore():
    """
    Test de Tietjen Moore. C'est une généralisation du test de Grubbs.
    """
    return
    
def Cook():
    return
    
def Peirce():
    return
    
def Mahalanobis():
    return
    


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
            

def supprime_un(x,v_poids,i,methode,sup_poids= 2,poids=1/100):
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

if __name__ == "__main__":
    #X,Y = ldt.load_points("droite_nulle_pasaberrant.txt") # Fonctionne avec Grubbs
    #X,Y = ldt.load_points("droite_nulle_un_aberrant.txt") # Fonctionne avec Grubbs
    #X,Y = ldt.load_points("droite_environ_nulle_pasaberrant.txt") # Fonctionne, mais tellement petits qu'un considéré comme aberrant : ça dépend de l'échelle !
    #X,Y = ldt.load_points("droite_environ_nulle_aberrant.txt") # Fonctionne avec Grubbs
    #X,Y = ldt.load_points("droite_identite.txt") # Fonctionne avec Grubbs
    #X,Y = ldt.load_points("droite_identite_environ_pasaberrant.txt") # Fonctionne avec Grubbs
    
    #X,Y = ldt.load_points("droite_identite_environ_aberrant.txt") # Ne fonctionne pas : problème du local/global. On devrait prendre ici les points 2 par 2 car peu denses
    
    
    res = Grubbs(Y)
    # Pour mes tests : rouge = aberrants, bleu = non aberrant
    x_rouge = []
    y_rouge = []
    x_bleu = []
    y_bleu = []
    for i in range(len(X)):
        if res[i] :
            x_rouge.append(X[i])
            y_rouge.append(Y[i])
        else :
            x_bleu.append(X[i])
            y_bleu.append(Y[i])
    plt.plot(x_rouge,y_rouge,color="red",linestyle = 'none',marker="o")
    plt.plot(x_bleu,y_bleu,color="blue",linestyle = 'none',marker="+")
