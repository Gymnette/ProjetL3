# -*- coding: utf-8 -*-

#####
# 1 #
#####
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

def Lk(x,xi,k):
    # x vecteur de 500 valeurs
    # xi vecteur des points d'interpolation
    # k-ième polynôme de Lagrange
    
    n = np.size(xi)
    y = np.ones(np.size(x))
    for i in range(n):
        if (i!=k):
            y *= (x - xi[i]) / (xi[k] - xi[i])
    return y

def ex1():   
    plt.figure()
    x = np.linspace(0,7,500)
    xi = np.linspace(0,7,8)

    for j in range(8):
        y = Lk(x,xi,j)
        nom = "k="+str(j)
        plt.plot(x,y,label = nom)
    
    plt.xlim(x[0], x[-1])
    plt.legend()

    plt.figure()
    x = np.linspace(-2,4,500)
    xa = np.linspace(-2,4,6)

    for j in range(6):
        y = Lk(x,xa,j)
        nom = "k="+str(j)
        plt.plot(x,y,label = nom)
    
    plt.xlim(x[0], x[-1])
    plt.legend()
    
    plt.figure()
    # on garde le même x
    xb = [-2,-1, -0.5, 0.5,3.5,4]

    for j in range(6):
        y = Lk(x,xb,j)
        nom = "k="+str(j)
        plt.plot(x,y,label = nom)
    
    plt.xlim(x[0], x[-1])
    plt.legend()

#####
# 2 #
#####

def f(x):
    return np.cos(1-x**2)*np.exp(-x**2+3*x-2)

def plotf():
    x = np.linspace(0,4,500)
    plt.figure()
    plt.plot(x,f(x),label = "test function f")
    
def ex2() : 
    plotf()

#####
# 3 #
#####

def interpolation(f,n,a,b):
    # f est la fonction que l'on veut interpoler
    # n est le degré du polynôme d'interpolation que l'on souhaite
    # [a,b] est l'intervalle sur lequel on veut interpoler f 
    px = 0
    x = np.linspace(a,b,500)
    xi = np.linspace(a,b,n+1)
    fxi = f(xi)
    for k in range(n+1):
        px += fxi[k] * Lk(x,xi,k)
    
    return x,px,xi,fxi
    
def ex3():
    #Interpolation de f par un polynôme de Lagrange de degré 8
    x,px,xi,fxi = interpolation(f,8,0,4) 
    plt.figure()
    plt.plot(x, px,label="uniform interpolant")
    plt.plot(x,f(x),"--",label="test function")
    plt.plot(xi,fxi,"o")
    plt.legend()

#####
# 4 #
#####

def ex4():
    plt.figure()
    #Affichage des polynômes de degré 2 à 20, par pas de 3, 
    #interpolant la fonction f
    a = 0
    b = 4
    for n in range(2, 21, 3):
        x,px,_,_ = interpolation(f,n,a,b)
        plt.plot(x, px, label="n = {}".format(n))
    plt.plot(x,f(x),"--k",label="test function")
    plt.legend()
    
#####
# 5 #
#####
    
def g(x):
    return 1/(1+25*x*x)

def ex5():
    plt.figure()
    #Affichage des polynômes de degré 2 à 20, par pas de 3, 
    #interpolant la fonction g
    a = -1
    b = 1
    for n in range(2, 21, 3):
        x,px,_,_ = interpolation(f,n,a,b)
        plt.plot(x,px,label="n = {}".format(n))
    plt.plot(x,f(x),"--k",label="test function")
    plt.legend()
    
ex5()