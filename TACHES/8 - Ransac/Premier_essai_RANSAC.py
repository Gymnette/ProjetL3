# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:09:26 2020

@author: Béryl
"""

import numpy as np
import matplotlib.pyplot as plt
from signaux_splines import *
from random import random,sample
import math
import load_tests as ldt

####################
# Fonctions utiles #
####################
    
def sortpoints(X,Y):
    """
    Fonction de tri de deux listes X et Y en fonction de X.
    Trie la liste X (sans modification globale) et range Y pour que son ordre corresponde
    au tri de X.
    
    Exemples :
    sortpoints([1,3,2,4],[5,6,7,8]) = [1,2,3,4],[5,7,6,8]
    """
    D = {}
    n = len(X)
    Xb = list(X)
    for i in range(n):
        D[X[i]] = Y[i]
    Xb.sort()
    Yb = [D[Xb[i]] for i in range(n)]
    return Xb,Yb

def alea(n,nb):
    """
    Génère nb entiers aléatoires dans [0,n[
    
    Type entrées :
        n : int
        nb : int
        
    Type sorties :
        list[int] de longueur nb
    """
    return sample(list(range(n)), nb)

def trouver_ind(elem,liste):
    """
    Renvoie l'indice de l'élément de liste le plus proche de elem
    
    Type entrées :
        elem : float
        liste : list[float]
    
    Type sorties :
        int
    """
    for i in range(len(liste)):
        if liste[i] == elem :
            return i
        elif liste[i] > elem :
            if i == 0 :
                return i
            elif liste[i] - elem < elem - liste[i-1]:
                return i
            else :
                return i-1
    return len(liste)-1
        

######################### 
# Fonctions de distance #
#########################
    
def d_euclidienne(x0,y0,x1,y1):
    """
    Calcul de la distance euclidienne entre (x0,y0) et (x1,y1)
    
    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float
    
    Type sorties :
        float
    """
    return math.sqrt((x0-x1)**2 + (y0 - y1)**2)
    

##############################################
# Fonctions de création de la spline cubique #
##############################################


def H0(t) :
    return 1 - 3 * t**2 + 2 * t**3
def H1(t) :
    return t - 2 * t**2 + t**3  
def H2(t) :
    return - t**2 + t**3
def H3(t) :
    return 3 * t**2 - 2 * t**3

def Matrix_NU(H):
    '''
    Renvoie la matrice associée au calcul des dérivées non uniformes
    '''
    n = len(H)
    Dm1 = [H[i] for i in range(1,n)]
    Dm1.append(1)
    Dm2 = np.array(Dm1)

    D1 =[2*(H[i-1]+H[i]) for i in range(1,n)]
    D1.append(2)
    D1.insert(0,2)
    D2 = np.array(D1)
    
    Dp1 = [H[i] for i in range(0,n-1)]
    Dp1.insert(0,1)
    Dp2 = np.array(Dp1)
    
    A = np.diag(D2)
    B = np.diag(Dm2,-1)
    C = np.diag(Dp2,1)

    return A+B+C
    
def Matrix_NU_resulat(Y,H):
    '''
    Renvoie le second membre du système de calcul des dérivées uniformes
    '''
    n = len(H)
    R = [3*(H[i-1]/H[i]*(Y[i+1]-Y[i])+H[i]/H[i-1]*(Y[i]-Y[i-1])) for i in range(1,n)]
    R.insert(0, 3/H[0]*(Y[1]-Y[0]))
    R.append(3/H[n-1]*(Y[n]-Y[n-1]))
    R = np.mat(R)
    return R.transpose()

def HermiteC1_non_affiche(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            x,y : cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    y = []
    for t in x:
        som = 0
        som+= y0*H0((t-x0)/(x1-x0))
        som+= y0p*(x1-x0)*H1((t-x0)/(x1-x0))
        som+= y1p*(x1-x0)*H2((t-x0)/(x1-x0))
        som+= y1*H3((t-x0)/(x1-x0))
        y.append(som)
    return(x,y)



def calcul_Spline_NU(X,Y,a,b,n):
    '''
    Calcul de la spline interpolant les données dans l'intervalle [a,b],
    sur n points d'interpolation non uniformes
    Si a et/ou b ne sont pas présents dans X, la spline est prolongée de manière linéaire entre a et min(x), et entre b et max(x)
    La pente vaut alors sur ces intervalles respectivement la dérivée au point min(x) et la dérivée au point max(x).
    Renvoie xres et yres, des vecteurs de réels donnant la discrétisation de la spline.
    '''
    H = [X[i+1]-X[i] for i in range(n-1)]
    #plt.scatter(X,Y,s=75,c='red',marker = 'o',label = "NU interpolation points")
    A = Matrix_NU(H)
    B = Matrix_NU_resulat(Y,H)
    Yp = np.linalg.solve(A,B)
    xres = []
    yres = []
    if X[0] != a :
        # y = cx+d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[0],Y[0])
        # On a alors y = YP[0] * x + (Y[0]-YP[0]X[0])
        xtemp = np.linspace(a,X[0],100)
        ytemp = [Yp[0] * x + (Y[0] - Yp[0]*X[0]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp
    
    for i in range(0,n-2):
        xtemp, ytemp = HermiteC1_non_affiche(X[i],Y[i],float(Yp[i]),X[i+1],Y[i+1],float(Yp[i+1]))
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        
        xres += xtemp
        yres += ytemp
    i = n-2
    xtemp, ytemp = HermiteC1_non_affiche(X[i],Y[i],float(Yp[i]),X[i+1],Y[i+1],float(Yp[i+1]))
    xtemp = list(xtemp)
    ytemp = list(ytemp)
    xres += xtemp
    yres += ytemp
    
    
    if X[-1] != b :
        # y = cx+d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[-1],Y[-1])
        # On a alors y = YP[-1] * x + (Y[-1]-YP[-1]X[-1])
        xtemp = np.linspace(X[-1],b,100)
        ytemp = [Yp[-1] * x + (Y[-1] - Yp[-1]*X[-1]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp
    
        
    return xres,yres

################################################
# Fonctions spécifiques aux splines de lissage #
################################################

# Création des matrices A,R,M,N pour trouver la matrice K 
    
def MatriceA(n,H): 
    """ 
    Création de la matrice A,  strictement diagonalement dominante et donc inversible
        
    Input : 
        n : entier(nombre de neouds)
        H : vecteur d'entiers (pas entre les entre xi)
    Output : 
        A : Matrice n,n (de flottants)
    """
    A=np.zeros((n,n))
    d=[2]
    d=np.append(d,[2*(H[i]+H[i+1]) for i in range(0,n-2)])
    d=np.append(d,2)
    dp1 = [1] + [H[i] for i in range(0,n-2)]
    dm1 = [H[i] for i in range(1,n-1)] + [1]
    A=np.diag(d)+np.diag(dm1,-1)+np.diag(dp1,1)
    return A

def MatriceR(n,H):
    """ 
    Création de la matrice R
    
    Input : 
        n : entier(nombre de neouds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        R : Matrice n,n (de flottants)
    """
    R=np.zeros((n,n))
    d=[-1/H[0]]
    d=np.append(d,[H[i]/H[i-1] - H[i-1]/H[i] for i in range(1,n-1)])
    d=np.append(d,1/H[n-2])
    dm1 = [-H[i]/H[i-1] for i in range(1,n-1)]
    dm1.append(-1/H[n-2])
    dp1 = [1/H[0]]
    dp1 = np.append(dp1,[H[i-1]/H[i] for i in range(1,n-1)])
    R=np.diag(d)+np.diag(dm1,-1)+np.diag(dp1,1)
    return 3.0*R

def MatriceM(n,H):
    """ 
    Création de la matrice M
    
    Input : 
        n : entier(nombre de noeuds)
        H : vecteur de flottants (pas entre les xi)
    Output : 
        M : Matrice n-2,n (de flottants)
    """
    M=np.zeros((n-2,n))
    for i in range(n-2):
        M[i][i]=1/H[i]**2
        M[i][i+1]=-(1/H[i]**2 + 1/H[i+1]**2)
        M[i][i+2]=1/H[i+1]**2
    return 3.0*M


def MatriceN(n,H):
    """ 
    Création de la matrice N
    
    Input :
        n : entier(nombre de noeuds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        N : Matrice n-2,n (de flottants)
    """
    N=np.zeros((n-2,n))
    for i in range(n-2):
        N[i][i]=1/H[i]
        N[i][i+1]=(2/H[i] - 2/H[i+1])
        N[i][i+2]=-1/H[i+1]
    return N


def MatriceK(n,H):
    """ 
    Création de la matrice K
    
    Input : 
        n : entier(nombre de neouds)
        H : vecteur de flottants (pas entre les xi)
    Output : 
        K : Matrice n,n (de flottants) 
    """
    return MatriceM(n,H) + (np.dot(np.dot(MatriceN(n,H),np.linalg.inv(MatriceA(n,H))),MatriceR(n,H)))


# Création des matrices H03,H12 pour trouver la matrice H 

def H03(N,n,uk,xi,H):
    """ 
    Création de la matrice HO3
    
    Input : 
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        H : vecteur de flottants (pas entre les xi)
    Output : 
        HO3 : Matrice n,n (de flottants)
    """
    M=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                M[j][i]=H0((uk[ki]-xi[i])/H[i])
                M[j][i+1]=H3((uk[ki]-xi[i])/H[i])
                j+=1
    return M


def H12(N,n,uk,xi,H):
    """ 
    Création de la matrice H12
    
    Input : 
        N : entier(taille de l'échantillon étudié), n - entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers, h - entier(pas régulier des noeuds sur l'intervalle du lissage [a,b])
        H : vecteur de flottants (pas entre les xi)
    Output:
        H12 : Matrice n,n (de flottants)
    """
    H12=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                H12[j][i]=H[i]*H1((uk[ki]-xi[i])/H[i])
                H12[j][i+1]=H[i]*H2((uk[ki]-xi[i])/H[i])
                j+=1
    return H12


def MatriceH(N,n,uk,xi,H):
    """ 
    Création de la matrice H
    
    Input : 
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        h : vecteur de flottants (pas entre les xi)
    Return :
        H : Matrice n,n (de flottants)
    """
    return H03(N,n,uk,xi,H) + (np.dot(np.dot(H12(N,n,uk,xi,H),np.linalg.inv(MatriceA(n,H))),MatriceR(n,H)))


# Création de la matrice S pour trouver la matrice W

def MatriceS(n,H):
    """ 
    Création de la matrice S
    
    Input : 
        n : entier(nombre de neouds)
        h : vecteur de flottants (pas entre les xi)
    Output :
        S : Matrice n,n de flottants
    """
    S=np.zeros((n-2,n-2))
    d= 2*np.array(H[0:n-2])
    dm1 = [1/2*H[i] for i in range(2,n-1)]
    dp1 = [1/2*H[i] for i in range(0,n-3)]
    S=np.diag(d)+np.diag(dm1,-1)+np.diag(dp1,1)
    return 1/3*S


def MatriceW(N,n,uk,xi,H,rho):
    """ 
    Création de la matrice W
    
    Intput : 
        uk : vecteur des abcisses (flottants) de l'échantillon étudié
        N : taille de uk (entier)
        xi : noeuds de lissage (flottants)
        n : taille de xi (entier)
        H : vecteur de flottants (pas entre les xi)
        rho :flottant,  paramètre de lissage qui contrôle le compromis entre la fidélité des données et le caractère théorique de la fonction
    Output : 
        W : matrice n,n de flottants
    """
    W1 = np.dot(np.transpose(MatriceH(N,n,uk,xi,H)),MatriceH(N,n,uk,xi,H))
    W2 = np.dot(np.dot(np.transpose(MatriceK(n,H)),MatriceS(n,H)),MatriceK(n,H))
    return W1 + (rho*W2)


def Matricew(zk,N,n,uk,xi,H):
    """ 
    Création de la matrice w
        
    Input : 
        uk,zk : vecteurs de float de l'échantillon étudié
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        xi : noeuds de lissage (flottants)
        H : vecteur de flottants (pas entre les xi)
    Output : 
        w : matrice n,n de flottants
    """
    return np.transpose(np.dot(zk,MatriceH(N,n,uk,xi,H)))

# Calcul du vecteur y
def Vecteur_y(uk,zk,N,xi,n,H,rho):
    """ 
    Création du vecteur y
    
    Intput : 
        uk,zk : vecteurs de float de l'échantillon étudié
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        xi : noeuds de lissage (flottants)
        H : vecteur de flottants (pas entre les xi)
        rho - float,  paramètre de lissage qui contrôle le compromis entre la fidélité des données et le caractère théorique de la fonction
    Output : y contenant la transposée des yi
    """
    return np.linalg.solve(MatriceW(N,n,uk,xi,H,rho),Matricew(zk,N,n,uk,xi,H))

def Repartition_chebyshev(a,b,n):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis façon chebyshev
    '''
    T = []
    t1 = float((a+b))/2
    t2 = float((b-a))/2
    for i in range (n):
        T.append(t1+t2*(np.cos((2*i+1)*np.pi/(2*n+2))))
    T.sort()
    T[0] = a
    T[-1] = b
    return T

def calcul_Spline_lissage(uk,zk,a,b,n,rho) :
    """ 
    Cette fonction calcule la spline de lissage des données uk zk sur l'intervalle [a,b].
    Elle considère n noeuds, répartis selon Chevyshev.
    Le paramètre de lissage est fourni à la fonction : rho.
    Renvoie la discrétisation de la spline de lissage.
    
    Type entrées :
        uk : vecteur de float
        zk : vecteur de float
        a : float
        b : float
        n : int
        rho : float
    
    Type sorties :
        list[float],list[float]
        
    """
    N = len(uk)
    xi = Repartition_chebyshev(a,b,n)
    
    
    H = [xi[i+1]-xi[i] for i in range(len(xi)-1)] # vecteur des pas de la spline
    
    Y = Vecteur_y(uk,[zk],N,xi,n,H,rho)
    
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(MatriceA(n,H),(np.dot(MatriceR(n,H),Y))))
    xx=[]
    yy=[]
    for i in range(n-1):
        x,y = HermiteC1_non_affiche(xi[i],yi[0][i],yip[0][i],xi[i+1],yi[0][i+1],yip[0][i+1])
        xx=np.append(xx,x)
        yy=np.append(yy,y)
    return xx,yy

##################################
# RANSAC : interpolation robuste #
##################################
# http://w3.mi.parisdescartes.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf 
# Nombre minimal de points ? Essai avec 3. Interpolation à chaque étape (et non approximation)
    
def ransac(x,y,nbitermax,err,dist,nbcorrect,nbpoints,rho):
    """
    Application de l'algorithme de Ransac avec un modèle de splines cubiques
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données, du nombre maximum d'itérations, 
    de l'erreur maximum acceptée entre un point et la spline, de la fonction distance associée, 
    du nombre de points nécessaires pour qu'un modèle soit considéré comme correct,
    du nombre de points pour créer la spline d'essai,
    et du paramètre de lissage
    xres et yres représentent la spline qui interpole au mieux les données d'après RANSAC

    Type entrées :
        x : vecteur de float
        y : vecteur de float
        nbitermax : int
        err : float
        dist : function : (float,float)x(float,float) -> float
        nbcorrect : int
        nbpoints : int
    Type sorties :
        xres : vecteur de float
        yres : vecteur de float
    """
    a = min(x)
    b = max(x)
    
    # Sauvegarde du meilleur modèle trouvé et de l'erreur associé
    xmod = None
    ymod = None
    errmod = -1
    
    for _ in range(nbitermax):
        # Choix d'un échantillon
        i_points = alea(len(x),nbpoints)
        x_selec = []
        y_selec = []
        for i in range(len(i_points)):
            x_selec.append(x[i_points[i]])
            y_selec.append(y[i_points[i]])
        
        # Création de la courbe à partir de l'échantillon
        x_selec,y_selec = sortpoints(x_selec,y_selec)
        xres,yres = calcul_Spline_NU(x_selec,y_selec,a,b,nbpoints)
        
        #plt.plot(x,y,"or")
        #plt.plot(x_selec,y_selec,"og")
        
        # Calcul des erreurs
        liste_inlier = list(i_points)
        for i in range(len(x)):
            if i in i_points :
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            else :
                i_associe = trouver_ind(x[i],xres)
                print(x[i],xres[i_associe])
                if dist(x[i],y[i],xres[i_associe],yres[i_associe]) <= err :                
                    liste_inlier.append(i)
                    #plt.plot(x[i],y[i],"bo")
        
        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier ! 
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            
            x_pour_spline = []
            y_pour_spline = []
            for i in range(len(liste_inlier)):
                x_pour_spline.append(x[liste_inlier[i]])
                y_pour_spline.append(y[liste_inlier[i]])
            x_pour_spline,y_pour_spline = sortpoints(x_pour_spline,y_pour_spline)
            xtemp, ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline,a,b,nbpoints,rho)
            err_temp = 0
            for i in range(len(x_pour_spline)):
                i_associe = trouver_ind(x_pour_spline[i],xtemp)
                err_temp += dist(x_pour_spline[i],y_pour_spline[i],xtemp[i_associe],ytemp[i_associe]) 
            if errmod == -1 or err_temp < errmod :
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)
    return xmod, ymod
        
def ransac_auto(x,y,err,dist,nbpoints,rho,pcorrect=0.99):
    """
    Automatisation de l'algorithme de Ransac avec un modèle de splines cubiques :
    le calcul de la proportion de points aberrants (outlier) ou non (inlier) est mise à jour au fur et à mesure.
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données,
    de l'erreur maximum acceptée entre un point et la spline, de la fonction distance associée,
    du nombre de points pour créer la spline d'essai,
    du paramètre de lissage,
    et de la probabilité d'obtenir un résultat correct avec cet algorithme.
    xres et yres représentent la spline qui interpole au mieux les données d'après RANSAC

    Type entrées :
        x : vecteur de float
        y : vecteur de float
        nbitermax : int
        err : float
        dist : function : (float,float)x(float,float) -> float
        nbcorrect : int
        nbpoints : int
    Type sorties :
        xres : vecteur de float
        yres : vecteur de float
    """
    a = min(x)
    b = max(x)
    
    prop_inlier = 0
    nbitermax = 10000 # Sera ajusté en fonction de prop_inlier
    nbcorrect = math.floor(prop_inlier*len(x))
    
    # Sauvegarde du meilleur modèle trouvé et de l'erreur associée
    xmod = None
    ymod = None
    errmod = -1

    k = 0
    while k < nbitermax :
        # Choix d'un échantillon
        i_points = alea(len(x),nbpoints)
        print(i_points)
        x_selec = []
        y_selec = []
        for i in range(len(i_points)):
            x_selec.append(x[i_points[i]])
            y_selec.append(y[i_points[i]])
        
        # Création de la courbe à partir de l'échantillon
        x_selec,y_selec = sortpoints(x_selec,y_selec)
        xres,yres = calcul_Spline_NU(x_selec,y_selec,a,b,nbpoints)
        
        #plt.plot(x,y,"or")
        #plt.plot(x_selec,y_selec,"og")
        
        # Calcul des erreurs
        liste_inlier = list(i_points)
        for i in range(len(x)):
            if i in i_points :
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            else :
                i_associe = trouver_ind(x[i],xres)
                if dist(x[i],y[i],xres[i_associe],yres[i_associe]) <= err :      
                    liste_inlier.append(i)
                    #plt.plot(x[i],y[i],"bo")
                #else :
                    #print(i)
                    #print(i_associe)
                    #print(dist(x[i],y[i],xres[i_associe],yres[i_associe]))
                    #print("fin")
        
        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier ! 
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            
            x_pour_spline = []
            y_pour_spline = []
            for i in range(len(liste_inlier)):
                x_pour_spline.append(x[liste_inlier[i]])
                y_pour_spline.append(y[liste_inlier[i]])
            x_pour_spline,y_pour_spline = sortpoints(x_pour_spline,y_pour_spline)
            xtemp, ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline,a,b,nbpoints,rho)
            #xtemp,ytemp = calcul_Spline_NU(x_pour_spline,y_pour_spline,a,b,len(x_pour_spline))
            # PARAMETRES :
            # Si la graine vaut 0 (signal stationnaire) : 0.001
            # Si elle vaut 1 ou 5 : 0.01
            # Pour le second signal stationnaire :
            # 0.0001 pour la graine qui vaut 0
            
            err_temp = 0
            for i in range(len(x_pour_spline)):
                i_associe = trouver_ind(x_pour_spline[i],xtemp)
                err_temp += dist(x_pour_spline[i],y_pour_spline[i],xtemp[i_associe],ytemp[i_associe]) 
            if errmod == -1 or err_temp < errmod :
                
                if not 1 in i_points and not 3 in i_points and not 7 in i_points :
                    print(i_points)
                    print("erreur diminuée !")
                    print(ymod)
                    print(ytemp)
                else :
                    print("erreur diminuée... :",err_temp,errmod)
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)
                
                #ESSAI D'AFFICHAGE DES POINTS ABERRANTS
                plt.close('all')
                plt.plot(x,y,"+b")
                plt.plot(x_pour_spline,y_pour_spline,"+y")
            
            # Que le modèle soit retenu ou non, on met à jour la proportion d'inliers et ce qui est associé 
   
            if prop_inlier < len(liste_inlier)/len(x):
                # Il y a plus d'inlier que ce qu'on pensait
                prop_inlier = len(liste_inlier)/len(x)
                if prop_inlier == 1 :
                    print("ici")
                    break; # On a trouvé un modèle correct, pour lequel il n'y aurait pas de points aberrants
                nbitermax = np.log(1 - pcorrect)/np.log(1-prop_inlier**len(x)) # Sera ajusté en fonction de prop_inlier
                nbcorrect = math.floor(prop_inlier*len(x))
        k += 1
    
    return xmod, ymod

def lancement_ransac(x,y,err,nconsidere,rho):
    x,y = sortpoints(x,y)
    xres,yres = ransac_auto(x,y,err,d_euclidienne,nconsidere,rho)
    plt.plot(xres,yres,"r")
        
if __name__ == "__main__":
    plt.close('all')
    plt.figure()
    
    ##########################################
    # Tests fonctionnels (paramètres réglés) #
    ##########################################
    
    # Utilisation : mettre le numéro de l'exemple ici. 0 <= num <= 9
    num = 1
    
    if num == 0: # Données de CAO
        plt.title("Ransac : données de CAO")
        x,y = np.loadtxt('data_CAO.txt')
        lancement_ransac(x,y,0.5,len(x)//10,0.1)
        xreel,yreel = calcul_Spline_lissage(x,y,min(x),max(x),len(x),0.1)
        plt.plot(xreel,yreel,"--b")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation obtenue","interpolation attendue"])
    elif num == 1 :
        plt.title("Ransac : droite environ nulle, données aberrantes")
        x,y = np.loadtxt('droite_environ_nulle_aberrant.txt')
        lancement_ransac(x,y,10,2,0.1)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation obtenue","interpolation attendue"])
        
    elif num == 2 :
        x,y = np.loadtxt('droite_environ_nulle_pasaberrant.txt')
    elif num == 3 :
        x,y = np.loadtxt('droite_identite.txt')
    elif num == 4 :
        x,y = np.loadtxt('droite_identite_environ_aberrant.txt')
    elif num == 5 :
        x,y = np.loadtxt('droite_identite_environ_pasaberrant.txt')
    elif num == 6 :
        x,y = np.loadtxt('droite_nulle_un_aberrant.txt')
    elif num == 7 :
        x,y = np.loadtxt('droite_nulle_pasaberrant.txt')
    elif num == 8:
        pass
    elif num == 9 :
        pass
    
    #nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
    
    #x,y = np.loadtxt('data_CAO.txt')
    
    # Seed sert à "fixer" le test
    #x,y, f = stationary_signal((30,), 0.9, noise_func=nfunc,seed=5) #Fonctionne bien pour 5 et 1, ainsi que 0 en fonction des paramètres
    
    #x,y, f = stationary_signal((30,), 0.5, noise_func=nfunc,seed=0)
    
    # Signaux non stationnaires
    x, y, f = non_stationary_signal((30,), switch_prob=0.1, noise_func=nfunc,seed=0)
    #x, y, f = non_stationary_signal((30,), switch_prob=0.2, noise_func=nfunc)
    
    
    #xspline,yspline = calcul_Spline_NU(x,y,a,b,n)
    #plt.plot(xspline, yspline,"--y")
    
    #xres,yres = ransac(x,y,20,0.5, d_euclidienne,n-(n//5),n//10)
    # Pour les signaux stationnaires générés avec la première commande : 0.5 d'erreur
    # Pour les seconds : 1 pour la graine 0
    #plt.plot(xres,f(xres))
    
    # Faire varier le dernier nombre ! (nombre de points sélectionnés pour créer un modèle)

    
   
   

