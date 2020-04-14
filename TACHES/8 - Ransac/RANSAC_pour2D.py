# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from signaux_splines import *
from random import random,sample
import math
import load_tests as ldt

import sys

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
        

def trouver_et_distance_ind_para(x,y,xc,yc,dist):
    """
    Renvoie la distance à la courbe de x,y, xc et yc représentant la courbe.
    
    Type entrées :
        x : float
        y : float
        xc : vecteur de float
        yc : vecteur de float
        dist : fonction : (float,float,float,float) -> float
    
    Type sorties :
        float
    """
    d = [dist(x,y,xc[i],yc[i]) for i in range(len(xc))]
    return min(d)

def calcul_erreur_courbe(xtest,ytest,xspline,yspline,dist):
    """
    Renvoie l'erreur totale des points à tester à la courbe, selon la fonction distance.
    
    Type entrées :
        xtest : vecteur de float
        ytest : vecteur de float
        xspline : vecteur de float
        yspline : vecteur de float
        dist : fonction : (float,float,float,float) -> float
        
    Type sorties :
        float
    """
    err = 0
    for i in range(len(xtest)):
        err += trouver_et_distance_ind_para(xtest[i],ytest[i],xspline,yspline,dist)
    return err

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
    
def d_carre(x0,y0,x1,y1):
    """
    Calcul de la distance euclidienne au carré entre (x0,y0) et (x1,y1)
    
    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float
    
    Type sorties :
        float
    """
    return d_euclidienne(x0,y0,x1,y1)**2

##############################################
# Fonctions de création de la spline cubique #
##############################################


def H0(t) :
    """
        # Renvoie H0 la première polynomiale cubique
        
        Input :  
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output : 
            H0 : flottant
    """
    return 1 - 3 * t**2 + 2 * t**3
def H1(t) :
    """
        # Renvoie H1 la deuxième polynomiale cubique
        
        Input :  
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output : 
            H1 : flottant
    """
    return t - 2 * t**2 + t**3  
def H2(t) :
    """
        # Renvoie H2 la troisième polynomiale cubique
        
        Input :  
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output : 
            H2 : flottant
    """
    return - t**2 + t**3
def H3(t) :
    """
        # Renvoie H3 la quatrième polynomiale cubique
        
        Input :  
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output : 
            H3 : flottant
    """
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
    
    R = []
        
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
    print("spline minx,maxx,lenx,miny,maxy,leny,a,b,n",min(X),max(X),len(X),min(Y),max(Y),len(Y),a,b,n)
    print("\n")
    H = [X[i+1]-X[i] for i in range(n-1)]
    
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

"""def H03(N,n,uk,xi,H):
    Création de la matrice HO3
    
    Input : 
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        H : vecteur de flottants (pas entre les xi)
    Output : 
        HO3 : Matrice n,n (de flottants)
    M=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                
                
                print("i,ki :",i,ki)
                print("j",j)
                print("N,n : ",N,n)
                print(len(uk),len(xi),len(H))
                print()
                
                
                M[j][i]=H0((uk[ki]-xi[i])/H[i])
                M[j][i+1]=H3((uk[ki]-xi[i])/H[i])
                j+=1
    return M"""
    

def Copie(c1,c2,i,j,M):
    # Copie les colonnes c1 et c2. Le premier élément de c1 est à l'indice i,j
    # c1 et c2 ont la même longueur.
    for k in range(len(c1)):
        M[i+k][j] = c1[k]
        M[i+k][j+1] = c2[k]
    return M

def H03(uk,xi,H):
    """
    Création de la matrice HO3
    
    Input : 
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        H : vecteur de flottants (pas entre les xi)
    Output : 
        HO3 : Matrice n,n (de flottants)
    """
    H03 = np.zeros((len(uk),len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi)-1) : #i de 1 à N-1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while (uk[k] < xi[i+1]):
            t = (uk[k]-xi[i])/H[i]# t^î k
            col1.append(H0(t))
            col2.append(H3(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1),1)
        c2 = np.array(col2).reshape(len(col2),1)
        H03 = Copie(c1,c2,kbase,i,H03) # Colonne i-1, ligne 
    return H03

def H12(uk,xi,H):
    """
    Création de la matrice H12
    
    Input :
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers, h - entier(pas régulier des noeuds sur l'intervalle du lissage [a,b])
        H : vecteur de flottants (pas entre les xi)
    Output:
        H12 : Matrice n,n (de flottants)
    """
    H12 = np.zeros((len(uk),len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi)-1) : #i de 1 à N-1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while (uk[k] < xi[i+1]):
            t = (uk[k]-xi[i])/H[i]# t^î k
            col1.append(H[i]*H1(t))
            col2.append(H[i]*H2(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1),1)
        c2 = np.array(col2).reshape(len(col2),1)
        H12 = Copie(c1,c2,kbase,i,H12) # Colonne i-1, ligne 
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
    return H03(uk,xi,H) + (np.dot(np.dot(H12(uk,xi,H),np.linalg.inv(MatriceA(n,H))),MatriceR(n,H)))


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

def Repartition_noeuds_donnees(x,nbnoeuds):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis de manière à avoir autant de données dans chaque intervalle.
    '''
    # nbnoeuds noeuds
    # <=> nbnoeuds-1 intervalles
    nbinter = nbnoeuds-1
    n = len(x)
    
    nbparinter = n//nbinter # Ne doit pas être nul, sinon problème.
    plus_un = n%nbinter
    # Les plus_un premiers intervalles auront un point de plus que nbparinter.
    noeuds = []
    compte = 0
    
    for i in range(n):
        if compte == 0 :
            noeuds.append(x[i])
            compte += 1
        else :
            if i < plus_un : #car i commence à 0, donc < et non <=
                if compte == nbparinter + 1 :
                    compte = 0
                else :
                    compte += 1
            else :
                if compte == nbparinter :
                    compte = 0
                else :
                    compte += 1
    noeuds.append(x[-1])
    #print("laaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    #plt.plot(noeuds,[0]*len(noeuds),"ob")
    return noeuds

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
    xi = Repartition_noeuds_donnees(uk,n)
    #print(uk,xi)
    
    #plt.plot(uk,zk,"+")
    #plt.plot(xi,[0]*n,"ob")
    
    
    
    
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
    #plt.plot(xx,yy)
    return xx,yy

def repartition_cordale(x,y,a,b,n):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis façon chordale
    '''
    D = [np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2) for i in range(n-2)]
    T = [a]
    som = np.sum(D)
    for i in range (n-2):
        T.append(T[i]+D[i]/som)
    T.append(b)
    T = (T-min(T)) / (max(T)-min(T))
    return T

def calcul_Spline_para_lissage(x,y):
    """
    Calcule la spline paramétrique cubique de lissage interpolant les données.
    Renvoie sa discrétisation.
    La paramétrisation est cordale.
    
    Type des entrées :
        x : vecteur de float
        y : vecteur de float
        a : int
        b : int
    
    Type des sorties :
        (vecteur de float, vecteur de float)
    """
    a = min(x)
    b = max(x)
    n = len(x)
    t =  repartition_cordale(x,y,a,b,n)

    _, xres = calcul_Spline_lissage(t,x,a,b,4,0.1)
    _, yres =calcul_Spline_lissage(t,y,a,b,4,0.1)

    return xres,yres

def calcul_Spline_para(x,y):
    a = min(x)
    b = max(x)
    n = len(x)
    t =  repartition_cordale(x,y,a,b,n)
    
    print("ICIIIIIIIIIIIIIIIIIIIIII",t)
        
    print("\n\n")

    t1, xres = calcul_Spline_NU(t,x,min(t),max(t),len(t))
    
    print(a,b)
    print(min(xres),max(xres))
    t2, yres =calcul_Spline_NU(t,y,min(t),max(t),len(t))
    
    plt.figure()
    plt.plot(t,x,"+")
    plt.plot(t1,xres)
    plt.figure()
    plt.plot(t,y,"+")
    plt.plot(t2,yres)
    plt.figure()
    plt.plot(x,y,"+")
    plt.plot(xres,yres)
    plt.figure()
    
    sys.exit()
    
    return xres,yres

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
        
def ransac_auto(x,y,err,dist,nbpoints,rho,pcorrect=0.99,para=False):
    """
    Automatisation de l'algorithme de Ransac avec un modèle de splines cubiques :
    le calcul de la proportion de points aberrants (outlier) ou non (inlier) est mise à jour au fur et à mesure.
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données,
    de l'erreur maximum acceptée entre un point et la spline, de la fonction distance associée,
    du nombre de points pour créer la spline d'essai,
    du paramètre de lissage,
    et de la probabilité d'obtenir un résultat correct avec cet algorithme.
    para = True si et seulement si on souhaite une spline paramétrique.
    xres et yres représentent la spline qui interpole au mieux les données d'après RANSAC
    si exact est vraie, tous les calculs sont effectués à partir de la spline cubique d'interpolation.

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
    nbitermax = 500 # Sera ajusté en fonction de prop_inlier
    nbcorrect = math.floor(prop_inlier*len(x))
    
    # Sauvegarde du meilleur modèle trouvé et de l'erreur associée
    xmod = None
    ymod = None
    errmod = -1

    k = 0
    deja_vu = []
    while k <= nbitermax :
        # Choix d'un échantillon
        i_points = alea(len(x),nbpoints)
        i_points.sort()
        # i_points contient toujours le même nombre de points distcints, il suffit de vérifier si ce sont les mêmes pour savoir si on a déjà fait ce cas.
        """fin = False
        for possibilite in deja_vu :
            fin = True
            for elem in i_points :
                if not elem in possibilite :
                    fin = False
                    break
            if fin :
                break
        if fin :
            k+=1
            continue # On a déjà étudié exactement ces indices là
        deja_vu.append(list(i_points))"""
        # JE NE SAIS PAS S'IL VAUT MIEUX TOUT CALCULER OU FAIRE UNE RECHERCHE A CHAQUE FOIS
        
        
        
        #print(i_points)
        x_selec = []
        y_selec = []
        for i in range(len(i_points)):
            x_selec.append(x[i_points[i]])
            y_selec.append(y[i_points[i]])
            
        # Création de la courbe à partir de l'échantillon
        if not para :
            x_selec,y_selec = sortpoints(x_selec,y_selec)
        xres,yres=[],[]
        if para :
            xres,yres = calcul_Spline_para(x_selec,y_selec)
            #plt.plot(x_selec,y_selec,"ro")
            #print(x_selec,y_selec)
            #plt.plot(xres,yres,"bo")
        else :
            xres,yres = calcul_Spline_NU(x_selec,y_selec,a,b,nbpoints)
        
        #plt.plot(x,y,"or")
        #plt.plot(x_selec,y_selec,"og")
        
        # Calcul des erreurs
        liste_inlier = list(i_points)
        #print()
        #print()
        #print("x et y :",x,"\n\n",y,"\n\n")
        for i in range(len(x)):
            if i in i_points :
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            else :
                i_associe = -1
                d_courbe = -1
                if para :
                    #print("appel trouverdistanceindpara (i,x[i],y[i],xres,yres)",i,x[i],y[i],xres,yres)
                    d_courbe = trouver_et_distance_ind_para(x[i],y[i],xres,yres,dist)
                else :
                    i_associe = trouver_ind(x[i],xres)
                    d_courbe = dist(x[i],y[i],xres[i_associe],yres[i_associe])
                if d_courbe <= err :      
                    liste_inlier.append(i)
                #else :
                    #print(i)
                    #print(i_associe)
                    #print(dist(x[i],y[i],xres[i_associe],yres[i_associe]))
                    #print("fin")
        
        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier ! 
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            liste_inlier.sort()
            x_pour_spline = []
            y_pour_spline = []
            for i in range(len(liste_inlier)):
                x_pour_spline.append(x[liste_inlier[i]])
                y_pour_spline.append(y[liste_inlier[i]])
            if not para :
                x_pour_spline,y_pour_spline = sortpoints(x_pour_spline,y_pour_spline)
            xtemp,ytemp = 0,0
            
            if para :
                xtemp,ytemp = calcul_Spline_para(x_pour_spline,y_pour_spline)
            else :
                xtemp,ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline,a,b,nbpoints,rho)
                        
            err_temp = -1
            if para :
                err_temp = calcul_erreur_courbe(x_pour_spline,y_pour_spline,xtemp,ytemp,dist)
                #print(i_points,err_temp)
            else :
                err_temp = 0
                for i in range(len(x_pour_spline)):
                    i_associe = trouver_ind(x_pour_spline[i],xtemp)
                    err_temp += dist(x_pour_spline[i],y_pour_spline[i],xtemp[i_associe],ytemp[i_associe]) 
            
            if errmod == -1 or err_temp < errmod :
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)
                
                #ESSAI D'AFFICHAGE DES POINTS ABERRANTS
                plt.figure()
                #plt.close('all')
                plt.plot(x,y,"+b")
                plt.plot(x_pour_spline,y_pour_spline,"+y")
                
                # AFFICHAGE DE LA SPLINE INTERMEDIAIRE
                # A LAISSER. DANS L'IDEE, OPTION A PROPOSER
                # Mais attention à bien modifier la légende
                #plt.plot(xres,yres,"--g")
            
            # Que le modèle soit retenu ou non, on met à jour la proportion d'inliers et ce qui est associé 
   
            if prop_inlier < len(liste_inlier)/len(x):
                # Il y a plus d'inlier que ce qu'on pensait
                prop_inlier = len(liste_inlier)/len(x)
                if prop_inlier == 1 :
                    break; # On a trouvé un modèle correct, pour lequel il n'y aurait pas de points aberrants
                nbitermax = math.floor(np.log(1 - pcorrect)/np.log(1-prop_inlier**len(x))) # Sera ajusté en fonction de prop_inlier
                if nbitermax > 500 :
                    nbitermax = 500
                nbcorrect = math.floor(prop_inlier*len(x))
        k += 1
    
    return xmod, ymod

def lancement_ransac(x,y,err,rho,nconsidere=-1):
    x,y = sortpoints(x,y)
    if (nconsidere == -1):
        nconsidere = len(x)//2
    xres,yres = ransac_auto(x,y,err,d_euclidienne,nconsidere,rho)
    plt.plot(xres,yres,"r")        
    
def lancement_ransac_para(x,y,err,rho,nconsidere=-1):
    if (nconsidere == -1):
        nconsidere = len(x)//2
    xres,yres = ransac_auto(x,y,err,d_euclidienne,nconsidere,rho,para=True)
    plt.plot(xres,yres,"r")  
        
if __name__ == "__main__":
    plt.close('all')
    plt.figure()
    
    ##########################################
    # Tests fonctionnels (paramètres réglés) #
    ##########################################
    
    # Utilisation : mettre le numéro de l'exemple ici. 0 <= num <= 31
    num = 32
    # 11 
    # 14 : paramètres pas trouvés
    
    # Données de CAO, nombreuses, sans points aberrants
    
    if num == 0: # Données de CAO
        x,y = np.loadtxt('data_CAO.txt')
        lancement_ransac(x,y,0.5,0.1)
        xreel,yreel = calcul_Spline_lissage(x,y,min(x),max(x),len(x),0.1)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : données de CAO")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
   
    # Petits tests spécifiques
    
    elif num == 1 : # Droite plus ou moins nulle, valeurs aberrantes
        x,y = np.loadtxt('droite_environ_nulle_aberrant.txt')
        lancement_ransac(x,y,2,5)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ nulle, données aberrantes")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 2 : # Droite plus ou moins nulle, pas de valeurs aberrantes
        x,y = np.loadtxt('droite_environ_nulle_pasaberrant.txt')
        lancement_ransac(x,y,0.2,0.1,nconsidere=8)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ nulle")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 3 : # Droite identité parfaite
        x,y = np.loadtxt('droite_identite.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite identité")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 4 : # Droite plus ou moins identité, valeurs aberrantes
        x,y = np.loadtxt('droite_identite_environ_aberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ identité, données aberrantes")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 5 : #Plus ou moins l'identité, sans valeurs aberrantes
        x,y = np.loadtxt('droite_identite_environ_pasaberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ identité")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 6 : # Droite nulle avec une donnée aberrante
        x,y = np.loadtxt('droite_nulle_un_aberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite nulle, donnée aberrante")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 7 :# Droite nulle sans données aberrantes
        x,y = np.loadtxt('droite_nulle_pasaberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite nulle")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    # Signaux stationnaires
    
    elif 8 <= num and num <= 13 :
        nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
        x,y, f = stationary_signal((30,), 0.9, noise_func=nfunc,seed=num-8)
        lancement_ransac(x,y,0.2,0.001)
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal stationnaire de régularité 0.9. seed = "+str(num-8))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif 14 <= num and num <= 19 :
        # JE N'ARRIVE PAS A TROUVER DE BONS PARAMETRES ICI
        nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
        x,y, f = stationary_signal((30,), 0.5, noise_func=nfunc,seed=num-14)
        if num == 19 :
            lancement_ransac(x,y,0.05,0.00001,nconsidere=26)
        elif num == 15 :
            lancement_ransac(x,y,0.2,0.00002,nconsidere=26)
        elif num == 16 :
            lancement_ransac(x,y,0.1,0.00001,nconsidere=20)
        elif num == 18 :
            lancement_ransac(x,y,0.5,0.00001,nconsidere=20)
        else :# 14 et 17
            lancement_ransac(x,y,0.05,0.00001,nconsidere=24)
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal stationnaire de régularité 0.5. seed = "+str(num-14))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    # Signaux non stationnaires    
    elif 20 <= num <= 25 :
        nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
        x, y, f = non_stationary_signal((30,), switch_prob=0.1, noise_func=nfunc,seed=num-20)
        if num == 21 :
            lancement_ransac(x,y,0.1,0.0005,nconsidere=20)
        else :
            lancement_ransac(x,y,0.2,0.00001)
            
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal non stationnaire avec une probabilité de saut de 0.1. seed = "+str(num-20))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    elif 26 <= num <= 31 : 
        nfunc = lambda x: add_bivariate_noise(x, 0.05, prob=0.15)
        x, y, f = non_stationary_signal((30,), switch_prob=0.2, noise_func=nfunc,seed=num-26)
        if num == 27:
            lancement_ransac(x,y,0.05,0.001,nconsidere=20) 
        if num == 29 or num == 30 or num == 31:
            lancement_ransac(x,y,0.1,0.00001,nconsidere=20) 
        else : 
            lancement_ransac(x,y,0.2,0.00001,nconsidere=25)   
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal non stationnaire avec une probabilité de saut de 0.1. seed = "+str(num-26))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])


    # PARAMETRIQUE
    elif num == 32 :
        x,y = np.loadtxt('2D2.txt')
        
        
        plt.figure()
        #plt.plot(x,y,"+")
        
        x = list(x)
        y = list(y)
        ind_a_suppr = []
        for i in range(1,len(x)):
            for elem in x[:i] :
                if x[i] < elem + 0.0001 and x[i] > elem - 0.0001:
                    ind_a_suppr.append(i)
                    break
        ind_a_suppr.reverse()
        for elem in ind_a_suppr :
            x.pop(elem)
            y.pop(elem)
            
        
        
        xreel = list(x)
        xreel.pop(8)
        yreel = list(y)
        yreel.pop(8)
        
        x.pop(len(x)-1)
        y.pop(len(y)-1)
        x.pop(len(x)-1)
        y.pop(len(y)-1)
        x.pop(len(x)-1)
        y.pop(len(y)-1)
        x.pop(len(x)-1)
        y.pop(len(y)-1)
        #lancement_ransac(x,y,2,1)
        lancement_ransac_para(x,y,2,1,nconsidere=len(x)//2)
        #plt.plot(xreel,yreel,"--b")
        #plt.title("Ransac : paramétrique")
        #plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
        
   

