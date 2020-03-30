# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def phi(x,omegai):
    return (np.sinh(omegai*x)-x*np.sinh(omegai)) / (np.sinh(omegai) - omegai)

def alphai(x,omegai):
    return (omegai*np.cosh(omegai*x)-np.sinh(omegai)) / (np.sinh(omegai) - omegai)

# basis from question 5 over [0,1] :
def B0(u) :
    return 1-u
def B1(u) :
    return u
def B2(u,omegai) :
    return phi(1-u)
def B3(u,omegai) :
    return phi(u)


def InterpoleB1(x0,y0,y0p,x1,y1,y1p,sigma,ss):
    """ interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = data of order 1 (real values)
        Return :
            plot the exponential interpolant
    """
    x = np.arange(x0,x1,ss)
    h = x1 - x0
    u = (x-x0)/h
    omega = sigma * h 
    alpha = alphai(u,omega)
    delta = (1+alpha) * (y1-y0) / h
    a = y0
    b = y1
    c = ((y0p * alpha + y1p - delta) * h) / (1-(alpha*alpha))
    d = ((delta - y1p * alpha - y0p ) * h) / (1 - (alpha*alpha))
    
    y = a * B0(u) + b * B1(u) + c * B2(u) + d * B3(u)
    plt.plot(x,y,'c--',lw=2,label='exponential interpolant')
    
    
    
def InterpoleB1Np(x0,y0,y0p,x1,y1,y1p,sigma,ss):
    """ interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = data of order 1 (real values)
        Return :
            the exponential interpolant
    """
    x = np.arange(x0,x1,ss)
    h = x1 - x0
    u = (x-x0)/h
    omega = sigma * h 
    alpha = alphai(u,omega)
    delta = (1+alpha) * (y1-y0) / h
    a = y0
    b = y1
    c = ((y0p * alpha + y1p - delta) * h) / (1-(alpha*alpha))
    d = ((delta - y1p * alpha - y0p ) * h) / (1 - (alpha*alpha))
    
    y = a * B0(u) + b * B1(u) + c * B2(u) + d * B3(u)
    return y

def tri(x,y):
    """
    trie x pa, et modifie y afin que les x et les y ayant les mêmes indices restent associés
    """
    lx = []
    ly = []
    for (a,b) in sorted([(x[i],y[i]) for i in range(len(x))]) :
        lx.append(a)
        ly.append(b)
    x = lx
    y = ly
        
    
def InterpolationSplineNonUniforme(a,b,sigmai,ss):
    plt.figure()
    plt.gca().set_xlim(a, b)
    plt.gca().set_ylim(a, b)
    xi,yi = PolygonAcquisition('oc','c--')
    tri(xi,yi) #On ne veut pas faire de paramétrique donc on veut que ce soit trié   
    n = len(xi)
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
        
    #DEFINIR M, matrice de la question 8 (nécessite les bornes)
    M =

    #DEFINIR res comme la matrice b de notre équation de la question 8
    res = 

    derivees = np.linalg.solve(M,np.transpose(np.matrix(res))) #Si res est déjà verticale, enlever la transposée. Si liste, la laisser
    for i in range(0,n-1):
        InterpoleB1(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],sigmai[i],ss)
        
    
def InterpolationSplineNonUniformeNp(a,b,n,xi,yi,ss):
    plt.figure()
    plt.gca().set_xlim(a, b)
    plt.gca().set_ylim(a, b)
    xi,yi = PolygonAcquisition('oc','c--')
    tri(xi,yi) #On ne veut pas faire de paramétrique donc on veut que ce soit trié   
    #COPIE COLLE DE LA FONCTION PRECEDENTE AVEC UNE MODIFICATION A LA FIN, UN RENVOI
    n = len(xi)
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
        
    #DEFINIR M, matrice de la question 8 (nécessite les bornes)
    M =

    #DEFINIR res comme la matrice b de notre équation de la question 8
    res = 

    derivees = np.linalg.solve(M,np.transpose(np.matrix(res))) #Si res est déjà verticale, enlever la transposée. Si liste, la laisser
    for i in range(0,n-1):
        y = InterpoleB1(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],sigmai[i],ss)
        for elem in y :
            yfinal.append(elem)
    return yfinal

    
def PolygonAcquisition(color1,color2) :
    """ Mouse acquisition of a polygon
        right click to stop
    """
    x = []  # x is an empty list
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            plt.plot(xx,yy,color1,markersize=8)
            x.append(xx)
            y.append(yy)
            plt.draw()
            if len(x) > 1 :
                plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    return x,y

def InterpolationParametrique(a,b,sigmai,ss):
    plt.figure()
    plt.gca().set_xlim(a, b)
    plt.gca().set_ylim(a, b)
    xi,yi = PolygonAcquisition('oc','c--')
    
    ti = np.linspace(a, b, len(xi)) #car a = 0 et b = 1
    
    di = [np.sqrt((yi[i+1]-yi[i])**2+(xi[i+1]-xi[i])**2) for i in range(len(xi)-1)]
    dt = np.sum(di)
    ti = [a]
    for i in range(len(di)):
        ti.append(ti[i]+(di[i]/dt))

    tx = InterpolationSplineNonUniformeNp(a,b,len(xi),ti,xi,ss)
    ty = InterpolationSplineNonUniformeNp(a,b,len(xi),ti,yi,ss)
    
    
    plt.plot(tx, ty,label='chordal parameterization')
    plt.legend()



    
    
    
