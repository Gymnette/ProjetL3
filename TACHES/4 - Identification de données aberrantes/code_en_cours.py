# -*- coding: utf-8 -*-
import load_tests as ldt
import scipy.stats as stat
from math import sqrt,floor
import matplotlib.pyplot as plt
from signaux_splines import *

####################
# Fonctions utiles #
####################

def moyenne(x,invisibles=None):
    """
    Cette fonction renvoie la moyenne des valeurs de x
    Les éléments à l'indice invisible sont ignorés
    
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
    Les éléments à l'indice invisible sont ignorés
    
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
    """
    Test de Grubbs.
    Grubbs est un cas particulier de la déviation extreme de Student.
    La fonction prend une liste de valeurs (ordonnées de points) et un paramètre alpha, le risque d'erreur qu'on accepte.
    L'algorithme de Grubbs est appliqué à la lettre : on applique la formule uniquement sur la valeur la plus éloignée.
    C'est pourquoi il faut appeler cette méthode tant qu'il y a un "vrai" dans le vecteur renvoyé.
    Elle renvoie une liste de booléens indiquant si la valeur associée est considérée comme aberrante selon le test de Grubbs.
    C'est le cas si la distance à la moyenne empirique est supérieure à un certain seuil.
    
    type des entrees :
        x : vecteur de float ou vecteur d'int
        alpha : float
        
    type des sorties :
        vecteur de booléens de la longueur de x
    """
    n = len(x)
    
    # Calculs de la moyenne et de l'écart type empiriques
    moy = moyenne(x)
    e_t = ecart_type(x,moy)
    
    if (e_t == 0): #Les valeurs sont toutes identiques, il n'y a pas de points aberrants
        return [False]*n
    
    # Calculs des distances à la moyennes, normalisées par l'écart type
    dist = [0]*n
    for i in range(n) :
        dist[i] = abs(x[i]-moy)
    
    dist = [d/e_t for d in dist]
    
    # Calcul de la distance limite
    tcrit = stat.t.ppf(1-(alpha/(2*n)),n-2)# Valeur critique selon la loi de Student avec n-2 degrés de liberté et une confiance de alpha/2N    
    dist_lim = (n-1)/sqrt(n) * sqrt(tcrit**2 / (n-2+tcrit**2))
    
    aberrant = [False]*n
    # On cherche la distance maximum avec son indice
    imax = 0
    dmax = 0
    for i in range(n):
        if dist[i] > dmax :
            dmax = dist[i]
            imax = i
    # Si cette distance est plus grande que la limite, la valeur est aberrante.
    aberrant[imax] = (dmax > dist_lim)
    return aberrant

# Le test de Tietjen Moore est une généralisation du test de Grubbs.
# Il peut être appliqué peu importe le nombre de valeurs aberrantes
# Mais il faut connaître ce nombre exactement : on n'implémente donc pas cette méthode.
    
def deviation_extreme_student(x,alpha=5/100, borne_max=0):
    """
    En anglais : extreme Studentized deviate (ESD)
    C'est la généralisation du test de Grubbs, sans avoir besoin d'itérer.
    D'après des études de Rosner (Rosner, Bernard (May 1983), Percentage Points for a Generalized ESD Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.) 
    , ce test est très précis pour n >= 25 et reste correct pour n>=15.
    Il faut donc faire attention : ne pas l'appeler sur un intervalle avec peu de points !
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
    #X,Y = ldt.load_points("droite_nulle_pasaberrant.txt") # Fonctionne avec Grubbs, ESD
    #X,Y = ldt.load_points("droite_nulle_un_aberrant.txt") # Fonctionne avec Grubbs, ESD
    #X,Y = ldt.load_points("droite_environ_nulle_pasaberrant.txt") # Fonctionne, mais tellement petits qu'un considéré comme aberrant : ça dépend de l'échelle !
    #X,Y = ldt.load_points("droite_environ_nulle_aberrant.txt") # Fonctionne avec Grubbs, ESD
    #X,Y = ldt.load_points("droite_identite.txt") # Fonctionne avec Grubbs, ESD
    #X,Y = ldt.load_points("droite_identite_environ_pasaberrant.txt") # Fonctionne avec Grubbs, EDS
    
    #X,Y = ldt.load_points("droite_identite_environ_aberrant.txt") # Ne fonctionne pas : problème du local/global. On devrait prendre ici les points 2 par 2 car peu denses
    
    # SIGNAUX DU PROF
    # definition de la fonction de bruit avec ecart type à 0.05 et probabilité de switch à 15%
    nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
    
    # signal stationnaire bruité de 30 points et régularité à 0.
    
    #X, Y, f = stationary_signal((30,), 0.9, noise_func=nfunc)
    # signal stationnaire bruité de 30 points et régularité à 0.5
    #X,Y, f = stationary_signal((30,), 0.5, noise_func=nfunc)

    #xi = np.linspace(0, 1, 100)
    #plt.plot(xi,f(xi))
    
    #res = Grubbs(Y)
    #res = Grubbs(Y,1/100)
    res = deviation_extreme_student(Y)
    #res = deviation_extreme_student(Y,borne_max = 1)
    
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
