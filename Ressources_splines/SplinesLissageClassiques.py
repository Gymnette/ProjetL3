# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Cubic Hermite basis over [0,1] :
def H0(t) :
    return 1 - 3 * t**2 + 2 * t**3
def H1(t) :
    return t - 2 * t**2 + t**3  
def H2(t) :
    return - t**2 + t**3
def H3(t) :
    return 3 * t**2 - 2 * t**3

def HermiteC1(x0,y0,y0p,x1,y1,y1p,color):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    t = (x-x0)/(x1-x0)
    y = y0 * H0(t) + y0p * (x1-x0) * H1(t) + y1p * (x1-x0) * H2(t) + y1 * H3(t)
    plt.plot(x,y,color,label='cubic interpolant')

def ConstructionA(n):
    A = np.diag((np.concatenate(([2], np.repeat(4,n-2),[2]))))
    A += np.diag(np.repeat(1,n-1),k=1)
    A += np.diag(np.repeat(1,n-1),k=-1)
    return A

def ConstructionR(n,h):
    R = np.diag((np.concatenate(([-1],np.zeros(n-2),[1]))))
    R += np.diag(np.repeat(1,n-1),k=1)
    R += np.diag(np.repeat(-1,n-1),k=-1)
    return (3.0/h) * R

def Copie(c1,c2,i,j,M):
    # Copie les colonnes c1 et c2. Le premier élément de c1 est à l'indice i,j
    # c1 et c2 ont la même longueur.
    for k in range(len(c1)):
        M[i+k][j] = c1[k]
        M[i+k][j+1] = c2[k]
    return M
    
def ConstructionH03(xi,uk,h):
    H03 = np.zeros((len(uk),len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi)-1) : #i de 1 à N-1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while (uk[k] < xi[i+1]):
            t = (uk[k]-xi[i])/h# t^î k
            col1.append(H0(t))
            col2.append(H3(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1),1)
        c2 = np.array(col2).reshape(len(col2),1)
        H03 = Copie(c1,c2,kbase,i,H03) # Colonne i-1, ligne 
    return H03
        
def ConstructionH12(xi,uk,h):
    H12 = np.zeros((len(uk),len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi)-1) : #i de 1 à N-1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while (uk[k] < xi[i+1]):
            t = (uk[k]-xi[i])/h# t^î k
            col1.append(H1(t))
            col2.append(H2(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1),1)
        c2 = np.array(col2).reshape(len(col2),1)
        H12 = Copie(c1,c2,kbase,i,H12) # Colonne i-1, ligne 
    return h*H12

def ConstructionM(h,n):
    M = np.zeros(((n-2),n))
    for i in range(n-2):
        M[i][i] = 1
        M[i][i+1] = -2
        M[i][i+2] = 1
    
    return (3.0/h**2.0) * M

def ConstructionN(h,n):
    N = np.zeros(((n-2),n))
    for i in range(n-2):
        N[i][i] = 1
        N[i][i+2] =- 1
    
    return (1.0/h) * N

def ConstructionS(h,n):
    S = np.diag((np.repeat(2,n-2))) + np.diag((np.repeat(0.5,n-3)),1)+ np.diag((np.repeat(0.5,n-3)),-1)
    return (h/3.0) * S

plt.close('all')
    
(uk,zk) = np.loadtxt('dataPasAberrante.txt') # Les points sont triés
plt.plot(uk,zk,"x")
a = uk[0]
b = uk[-1]
N = len(uk)
#MODIFIABLES POUR LES TESTS
n = 15 #Nombre de points d'interpolation xi,yi
#############
#On veut y et y', afin de pouvoir tracer la courbe à l'aide de HermiteC1 interpôle les (xi,yi)

xi = np.linspace(a,b, n)
h = xi[1]-xi[0] #Taille d'un intervalle entre xi , car les xi sont répartis uniformément

Am1 = np.linalg.inv(ConstructionA(n))
R = ConstructionR(n,h)

H03 = ConstructionH03(xi,uk,h)
H12 = ConstructionH12(xi,uk,h)
H = H03+H12.dot(Am1).dot(R)
M = ConstructionM(h,n)
N = ConstructionN(h,n)

K = M + N.dot(Am1).dot(R)
S = ConstructionS(h,n)
wt = np.transpose(zk).dot(H)
w = np.transpose(wt)

couleurs = ["k","b","c","m","r","g"]
c = 0
for rho in [0.001,0.1,1,10,100,10000]:
    color = couleurs[c]
    c+=1
    W = np.transpose(H).dot(H) + rho * np.transpose(K).dot(S).dot(K)
    y = np.linalg.solve(W,w)#Wy = w
    #Calcul de y'
    yprime = Am1.dot(R).dot(y)

    #Traçage de la courbe
    for i in range(len(xi)-1):
        HermiteC1(xi[i],y[i],yprime[i],xi[i+1],y[i+1],yprime[i+1],color)



