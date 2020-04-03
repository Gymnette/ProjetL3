# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:43:04 2020

@author: zakaria
"""


import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stat

def quartile(x,i,Q=0.01):
    """
    cette fonction prend un vecteur x, un indice i et un parametre Q
    et qui revoie True si le pois x[i] est un point aberrant, et 
    renvoie False sinon
    """
    x_s = sorted(x)
    n = len(x_s)
    if n <3 :
        return True
    elif n == 3 :
        return False
    else:
        
        k = n//4 
        """
        if (n/4)-(n//4) != 0 :
            k =(n//4) +1
        else:
            k =n//4 +1
        """
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
    cette fonction prend un vecteur x et un indice i 
    et qui revoie True si le pois x[i] est consideré comme un point
    aberrant selon le test de Chauvenet, et renvoie False sinon
    """
    n = len(x) 
    x_barre = sum(x)/n
    var_x = (1/n)*sum(np.array(x)**2) - x_barre**2
    #print("var_x = ", var_x**0.5)
    a = abs(x[i]-x_barre)/var_x**(0.5)
    n_a = 2*(1-stat.norm.cdf(a,loc = 0,scale = 1))
    #print("n_a = ", n_a)
    if n_a < 0.5 :
        return True
    else :
        return False
    
def thompson(x,i,alpha=0.001):
    """
    cette fonction prend un vecteur x, un indice i et un parametre alpha
    et qui revoie True si le pois x[i] est un point aberrant, et 
    renvoie False sinon
    """
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
        

def supp_aberr(x,y,M=1) :
    """
    cette foction supprime les points (xi,yi) s'ils sont considéré comment 
    des points aberrants
    le parametre M prend trois valeurs {1,2,3}, 1 si on veut utiliser 
    la méthode de Chauvenet, 2 la méthode de thompson, 3 la méthode 
    d'inter-quartile
    """
    x_d = []
    y_d = []
    
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
 
def pas_inter(y,epsilon=0.1):
    """
    cette fonction prend un vecteur y et un parametre de variation epsilon
    et qui renvoie des intervalles sur lesquels la variation de y est 
    inferieure à epsilon
    """
    p = [0]
    n = len(y)
    for i in range(n-2):
        d_yi = y[i+1]-y[i]
        d_yi_1 = y[i+2]-y[i+1]
        delta = abs(d_yi - d_yi_1)
        
        if delta > epsilon :
           # c +=1
            p.append(i+1)
            
    p.append(n-1)       

    return p   


if __name__ == "__main__" :

    (uk,uz) = np.loadtxt('data.txt')
    
    """
    _________CAS_NON_UNIFORME__________
    """
    
    n =len(uk)
    p = pas_inter(uz,epsilon=0.075)
    a = p[0]
    b = p[1]
    X = []
    Y = []
    M = 1
    i=2
    while i < len(p) :
        j = uk[a:b]
        g = uz[a:b]
        
        xd, yd = supp_aberr(j,g,M)
        X = X + xd
        Y = Y + yd

        a = b
        b = p[i]
        i+=1
        


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
        

    


       
