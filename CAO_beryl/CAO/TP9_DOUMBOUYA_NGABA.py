#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:40:27 2019

@author: doumbomo

    DOUMBOUYA Mohamed & NGABA Billy
    
"""

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

# Cubic Hermite C1 interpolation over 2 points
def HermiteC1(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    
    y = y0*H0((x-x0)/(x1-x0)) + y0p*(x1-x0)*H1((x-x0)/(x1-x0)) + y1p*(x1-x0)*H2((x-x0)/(x1-x0)) + y1*H3((x-x0)/(x1-x0)) 
    
    return x,y


# Création des matrices A,R,M,N pour trouver la matrice K 
def MatriceA(n):
    M=np.zeros((n,n))
    d=[2]
    d=np.append(d,4*np.ones(n-2))
    d=np.append(d,2)
    M=np.diag(d)+np.diag(np.ones(n-1),-1)+np.diag(np.ones(n-1),1)
    return M

def MatriceR(n,h):
    M=np.zeros((n,n))
    d=[-1]
    d=np.append(d,np.zeros(n-2))
    d=np.append(d,1)
    M=np.diag(d)+np.diag(-np.ones(n-1),-1)+np.diag(np.ones (n-1),1)
    return (3.0/h)*M

def MatriceM(n,h):
    M=np.zeros((n-2,n))
    for i in range(n-2):
        M[i][i]=1
        M[i][i+1]=-2
        M[i][i+2]=1
    return (3.0/(h**2))*M

def MatriceN(n,h):
    M=np.zeros((n-2,n))
    for i in range(n-2):
        M[i][i]=1
        M[i][i+2]=-1
    return (1.0/h)*M

def MatriceK(n,h):
    return MatriceM(n,h) + (np.dot(np.dot(MatriceN(n,h),np.linalg.inv(MatriceA(n))),MatriceR(n,h)))

# Création des matrices H03,H12 pour trouver la matrice H 
def H03(N,n,uk,xi,h):
    M=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                M[j][i]=H0((uk[ki]-xi[i])/h)
                M[j][i+1]=H3((uk[ki]-xi[i])/h)
                j+=1
    return M

def H12(N,n,uk,xi,h):
    M=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                M[j][i]=H1((uk[ki]-xi[i])/h)
                M[j][i+1]=H2((uk[ki]-xi[i])/h)
                j+=1
    return h*M

def MatriceH(N,n,uk,xi,h):
    return H03(N,n,uk,xi,h) + (np.dot(np.dot(H12(N,n,uk,xi,h),np.linalg.inv(MatriceA(n))),MatriceR(n,h)))

# Création de la matrice S pour trouver la matrice W
def MatriceS(n,h):
    M=np.zeros((n-2,n-2))
    d=[2]
    d=np.append(d,2*np.ones(n-4))
    d=np.append(d,2)
    M=np.diag(d)+np.diag((1/2)*np.ones(n-3),-1)+np.diag((1/2)*np.ones(n-3),1)
    return (h/3.0)*M

def MatriceW(N,n,uk,xi,h,rho):
    Temp1 = np.dot(np.transpose(MatriceH(N,n,uk,xi,h)),MatriceH(N,n,uk,xi,h))
    Temp2 = np.dot(np.dot(np.transpose(MatriceK(n,h)),MatriceS(n,h)),MatriceK(n,h))
    return Temp1 + (rho*Temp2)

# Calcul de la matrice w
def Matricew(zk,N,n,uk,xi,h):
    return np.transpose(np.dot(zk,MatriceH(N,n,uk,xi,h)))

# Calcul du vecteur y
def Vecteur_y(uk,zk,N,xi,n,h,rho):
    """
        Entrées : - (uk,zk,N:leur taille)
                  - (xi,n:sa taille et celle de yi (le résultat))
                  - (h,rho)
        Sortie : y contenant la transposée des yi
    """
    return np.linalg.solve(MatriceW(N,n,uk,xi,h,rho),Matricew(zk,N,n,uk,xi,h))

# Création des matrices pour trouver y'
def Matdiag(n):
    M=np.zeros((n,n))
    d=[2]
    d=np.append(d,4*np.ones(n-2))
    d=np.append(d,2)
    M=np.diag(d)+np.diag(np.ones(n-1),-1)+np.diag(np.ones(n-1),1)
    return M

def transforme_zk_en_matrice(zk):
    return [zk]
            

"""------------------------------------------------------
MAIN PROGRAM :    #yip = Interpolation_Spline_Uniforme(yi[0],h,n)
------------------------------------------------------"""

# Les données
(uk,zk)=np.loadtxt('data.txt')
plt.plot(uk,zk,'rx',label='scattered data')
N = len(uk)

n=15
plt.title('smoothing spline with '+str(n)+' knots')
a = -2
b = 8
xi = np.linspace(a,b,n)
h = (b-a)/(n-1)
rho = [0.001,0.1,1.0,10.0,100.0,10000.0]

for j in range(len(rho)):
    Y = Vecteur_y(uk,transforme_zk_en_matrice(zk),N,xi,n,h,rho[j])
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(MatriceA(n),(np.dot(MatriceR(n,h),Y))))
    xx=[]
    yy=[]
    for i in range(n-1):
        x,y = HermiteC1(xi[i],yi[0][i],yip[0][i],xi[i+1],yi[0][i+1],yip[0][i+1])
        xx=np.append(xx,x)
        yy=np.append(yy,y)
    plt.plot(xx,yy,lw=1,label='smoothing spline with rho = '+str(rho[j]))

#plt.legend(loc='best')
