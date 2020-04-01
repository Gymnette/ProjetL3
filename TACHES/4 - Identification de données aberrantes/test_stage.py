# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:43:04 2020

@author: zakaria
"""


import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stat

def quartile(x,i,Q=0.1):
    """
    la fonction prend un vecteur et qui renvoie un intervalle [a,b]
    un point est considéré aberrant s'il n'appartient pas à intervalle [a,b]
    """
    x_s = sorted(x)
    n = len(x_s)
    k = n/4
    if (n/4)-int(n/4) != 0 :
        k =int(n/4) +1
    else:
        k =n//4 +1
    
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
    test de Chauvenet
    """
    n = len(x) 
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    #print("var_x = ", var_x**0.5)
    a = abs(x[i]-x_barre)/var_x**(0.5)
    n_a = (2*stat.norm.cdf(a,loc = 0,scale = 1)-1)
    #print("n_a = ", n_a)
    if n_a > 0.5 :
        return True
    else :
        return False
    
def thompson(x,i,alpha=0.01):
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
        

"""
la fonction supprime prend un vecteur x, un indice i, le nom une methode de
detection des points aberrants, un booléen sup_poid égale à 1 si on veut supprimer
les points aberrants,égale à 2 si on veut garder la taille de x inchangée (None au lieu de points aberrants)
égale à 3 si on veut remplacer les points aberrants par les valeurs des quantiles
et égale à 0 si on veut affecter le poid "poid" aux pointx 
aberrants et un poid = 1 aux points normaux 
"""
def supprime_un(x,v_poids,i,methode,sup_poid= 2,poids=1/100):
    a,b = methode(x)
    xk = x[i]
    x_sup = list(x)
    
    if xk <a or xk>b:
        if sup_poid == 1:
            x_sup.pop(i)
            v_poids.pop(i)
        elif sup_poid == 2:
            x_sup[i] = None
            v_poids[i] = None
        elif sup_poid == 3:
            x_sup[i] = a if xk<a else b
        else:
            v_poids[i] = poids

    return x_sup,v_poids

#def local_global(x):
def supp_aberr(x,y,M=1) :
    
    x_d =[]
    y_d = []
    N = (np.array(x)**2 + np.array(y)**2)**0.5
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
    

if __name__ == "__main__" :

    x = [5,7,10,15,19,21,21,22,23,23,23,23,23,24,24,24,24,25]
    #print(x)
    v_poids = [1]*len(x)
    (uk,uz) = np.loadtxt('data.txt')

    """
    i = 0
    n = len(x)
    while i<n :
        x,v_poids = supprime_un(x,v_poids,i,quartile,sup_poid= 1,poids=1/100)
        if len(x) == n:
            i+=1
        n = len(x)
    print(x)
    """
    """
    x_ecart = []
    y_ecart = []
    x_thomp = []
    y_thomp = []
    #N = (np.array(uk)**2 + np.array(uz)**2)**0.5
    N = abs(np.array(uk) - np.array(uz))
    print("x = ", x)
    
    for i in range(len(uk)):
        
        if test_Chauvenet(uz,i) == False :
            x_ecart.append(uk[i])
            y_ecart.append(uz[i])
        
        if thompson(N,i) == False :
            x_thomp.append(uk[i])
            y_thomp.append(uz[i])
           
        if quartile(N,i) == False :
            x_thomp.append(uk[i])
            y_thomp.append(uz[i])
            
    plt.plot(x_thomp,y_thomp,'or',color='r')
        
    print("x_ecart = ",x_ecart)     
    print("x_thomp = ",x_thomp)   
    """    
        
    n =len(uk)
    k = 0
    pas = n//20
    a = 0
    b = pas
    X = []
    Y = []
    M = 3
    while b <= n :
        j = uk[a:b]
        g = uz[a:b]
        
        xd, yd = supp_aberr(j,g,M)
        X = X + xd
        Y = Y + yd

        a = b
        b += pas
    if M==1:
        lab = "chauvenet"
    elif M==2 :
        lab ="thompson"
    elif M==3 :
        lab ="quartile"
        
    plt.figure(lab)
    plt.plot(uk,uz,'rx',color='b',label="données aberrantes")
    plt.plot(X,Y,'or',color='r',label="données non aberrantes")
    plt.legend(loc='best')

    x_d1 =[]
    x_d2 = []
    x_d3 = []
    
    for i in range(len(x)):
        if test_Chauvenet(x,i) == False :
            x_d1.append(x[i])
                
        if thompson(x,i) == False :
            x_d2.append(x[i])
                
        if quartile(x,i) == False :
            x_d3.append(x[i])
                
    print('x= ', x)
    print('Chauvenet = ', x_d1)
    print('thompson = ', x_d2)
    print('quartile = ', x_d3)





       
