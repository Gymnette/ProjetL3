#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:19:20 2020

@author: Amelys Rodet
"""

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import doctest
import warnings as wrng

"""
Examples Functions
"""
def f2dtab(X,Y):
    n  = len(X)
    Z = np.zeros((n,n))
    for i in range(len(X)):
        x = X[i]
        for j in range(len(Y)):
            y = Y[j]
            if x**2+y**2<=1:
                Z[i][j]= np.sqrt(1-x**2-y**2)
            else:
                Z[i][j]= 0
    return Z

def identity(x):
    return x

"""
Tool functions
"""

def isarraylike(X):
    return isinstance(X,np.ndarray) or isinstance(X,list)

def plot1d2d(T, fx, fy, legend="", new_fig=True, show=True,color = 'r',title = ""):
    """
    3d parametric curve
    
    Aguments :
    :type T: array (dim = 1)
    :type fx: function (fx(t) = x[t]) OR 1D numpy array
    :type fy: function (fy(t) = y[t]) OR 1D numpy array
    
    Optional arguments :
    :type legend: string (graph legend)
    :type new_fig: boolean (draws on an existing figure if False)
    :type show: boolean (shows the graphic window if True)
    
    :return: None
    
    :raises: Warning if fx and fy are not of the same type. User might be mistaking.
    
    :Examples:
        
    >>> plot1d2d(np.linspace(0,10,50),identity,identity,show = False)
    
    >>> plot1d2d(np.linspace(0,10,50),np.sin,np.cos,show = False)
    
    >>> T = np.linspace(0,10,50)
    >>> plot1d2d(T,np.cos(T),np.sin(T),new_fig = False)
    
    >>> plot1d2d(T,np.cos(T),np.sin)
    
    """
    if (isarraylike(fx) and not isarraylike(fy)) or (isarraylike(fy) and not isarraylike(fx)):
        wrng.warn("function called with a function and an array. Is it normal ?")
    
    if isarraylike(fx):
        X = fx
    else:
        X = fx(T)
        
    if isarraylike(fy):
        Y = fy
    else:
        Y = fy(T)
        
    if new_fig :
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if title != "":        
            fig.canvas.set_window_title(title)
    else:
        ax = plt.gca(projection = '3d')
    if legend == "":
        ax.plot(T, X, Y)
    else :
        ax.plot(T, X, Y, label=legend,c = color)
        ax.legend()

    if show:
        plt.show()
    
    return
    

def plot1d1d(X,Y,c = 'r',legend = "",title = "",new_fig = True,show = True,xbounds = [None,None],ybounds = [None,None], xlabel=None, ylabel=None):
    """
    2D curve plot
    
    Arguments :
            X,Y des tableaux 1d (souvent Y = f(X))
        Sortie : 
            Affiche une fenêtre 3d
            
    Aguments :
    :type X: 1D numpy array
    :type Y: function (Y(xt) = y[xt]) OR 1D numpy array
    
    Optional arguments :
    :type legend: string (graph legend)
    :type newfig: boolean (draws on an existing figure if False)
    :type show: boolean (shows the graphic window if True)
    :type xbounds: List of 2 numbers [xmin,xmax]
    :type ybounds: List of 2 numbers [ymin,ymax]
    
    :return: None
    
    :raises: Nothing
    
    :Examples:
        
    >>> plot1d1d(np.linspace(0,10,50),identity,show = False)
    
    >>> plot1d1d(np.linspace(0,10,50),np.sin,show = False)
    
    >>> T = np.linspace(0,10,50)
    >>> plot1d1d(T,np.cos(T),new_fig = False)
    """

    
    if not (isarraylike(Y)):
        Y = Y(X)
        
    if new_fig:
        fig = plt.figure()
        ax = fig.gca()
        if title != "":        
            fig.canvas.set_window_title(title)
    else:
        ax = plt.gca()
    if xbounds != [None,None]:
        ax.set_xlim(xmin=xbounds[0], xmax = xbounds[1])
    if ybounds != [None,None]:
        ax.set_ylim(ymin=ybounds[0], ymax = ybounds[1])
        
    if legend == "":
        plt.plot(X,Y,color = c)
    else : 
        plt.plot(X,Y,color = c,label = legend)
        ax.legend()

    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)

    if show :
        plt.show()
        
    return

def plot2d1d(X,Y,fz,c = cm.Oranges,title = "", show=True,show_scale =True) :
    """
    Plot d'un champ scalaire
        Arguments :
            X,Y des tableaux 1d
            f fonction de R2 dans R
        Sortie :
            Affiche une fenêtre 3d
    """
    #Z = fz(X,Y)
    n  = len(X)
    if not (isarraylike(fz)):
        Z = np.zeros((n,n))
        for i in range(len(X)):
            x = X[i]
            for j in range(len(Y)):
                y = Y[j]
                Z[i][j]= fz(x,y)
        Xm, Ym = np.meshgrid(X, Y)
    else:
        Z = fz
        Xm, Ym = X,Y
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(Xm, Ym, Z, cmap=c,linewidth=0, antialiased=False)
    if show_scale:
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if title != "":
        fig.canvas.set_window_title(title)

    if show:
        plt.show()
    
def plot3d3d(X,Y,Z,f, show=True):
    """
    Plot d'un champ vectoriel 3d
    
        Arguments :
            X,Y,Z des tableaux 1d
            f(X,Y,Z)
        Sortie : 
            Affiche une fenêtre 3d
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    U,V,W = f(X,Y,Z)
    # Make the grid
    x, y, z = np.meshgrid(X,Y,Z)
    
    # Make the direction data for the arrows
    u,v,w = np.meshgrid(U,V,W)
    
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

    if show: 
        plt.show()


def plot2d2d(X, Y, f, space=3, color='b', new_fig=True, show=True, xlabel=None, ylabel=None):
    """
    Plot d'un champ vectoriel 2d
    
        Arguments :
            X,Y,Z des tableaux 1d
            f(X,Y,Z)
            color : couleur (optionel)
        Sortie : 
            Affiche une fenêtre 2d
    """
    
    if not isarraylike(f):
        U,V = f(X,Y)
    else:
        U = [e[0] for e in f]
        V = [e[1] for e in f]
    # Make the grid
    x, y = np.meshgrid(X,Y)
    # Make the direction data for the arrows
    u,v = np.meshgrid(U,V)
    if new_fig:
        fig, ax = plt.subplots()
        ax.quiver(x[::space, ::space], y[::space, ::space], u.transpose()[::space, ::space], v.transpose()[::space, ::space],color = color)
    else:
        plt.quiver(x[::space, ::space], y[::space, ::space], u.transpose()[::space, ::space], v.transpose()[::space, ::space],color = color)
    
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    if show: 
        plt.show()
    
def plotlevel(X, Y, fz, tab_levels=None, c=cm.CMRmap, title="", new_fig=True, show=True):
    """
    Plot d'une courbe de niveau
    
        Arguments :
            X,Y des tableaux 1d
            f fonction de R2 dans R
        Sortie : 
            Affiche une fenêtre 3d
    """
    
    n  = len(X)
    Z = np.zeros((n,n))
    for i in range(len(X)):
        x = X[i]
        for j in range(len(Y)):
            y = Y[j]
            Z[i][j]= fz(x,y)
    Xm, Ym = np.meshgrid(X, Y)
    
    # Plot the surface.
    if new_fig :
        plt.figure()

    contours = plt.contour(X,Y,Z,levels = tab_levels)
    plt.clabel(contours, inline=1, fontsize=10)
    plt.title(title)

    if show:
        plt.show()

if __name__ == "__main__":
    All_functions = {0:('parametric curve', plot1d2d),1: ('one dimension',plot1d1d),2 : ('scalar field',plot2d1d), 3 : ("vectorial field '3D", plot3d3d), 4 : ("vectorial field (2D)", plot2d2d), 5 : ("leveling curves", plotlevel)}
    
    print("which running exemple ?")
    for i in range(len(All_functions)):
        print(i,' :',All_functions[i][0])
    ex = int(input("\n"))
    help(All_functions[ex][1])
    doctest.run_docstring_examples(All_functions[ex][1],globals())
        
