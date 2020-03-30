# -*- coding: utf-8 -*-
"""
BIICHLE Dorian
GUERY Alexandre
PIASENTIN Béryl
"""

###############---BIBLIOTHEQUES---#####################
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

###############---FONCTIONS---#####################
def phi(x,omegai):
    # Calcul de phi i(x), x étant entre 0 et 1
    return (np.sinh(omegai*x)-x*np.sinh(omegai)) / (np.sinh(omegai) - omegai)

def alphai(omegai):
    # Calcul de alphai = phi'(1)
    return (omegai*np.cosh(omegai*1)-np.sinh(omegai)) / (np.sinh(omegai) - omegai)

def betai(omegai):
    # Calcul de betai = phi"(1)
    return (omegai*omegai*np.sinh(omegai*1))/(np.sinh(omegai) - omegai)

# base de la question 5 pour u dans [0,1] :
def B0(u) :
    return 1-u
def B1(u) :
    return u
def B2(u,omegai) :
    return phi(1-u,omegai)
def B3(u,omegai) :
    return phi(u,omegai)



def InterpoleB1(x0,y0,y0p,x1,y1,y1p,sigma,ss,color):
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
    alpha = alphai(omega)
    delta = (1+alpha) * (y1-y0) / h
    a = y0
    b = y1
    c = ((y0p * alpha + y1p - delta) * h) / (1-(alpha*alpha))
    d = ((delta - y1p * alpha - y0p ) * h) / (1 - (alpha*alpha))
    
    y = a * B0(u) + b * B1(u) + c * B2(u,omega) + d * B3(u,omega)
    plt.plot(x,y,color,lw=2)
    
    
    
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
    alpha = alphai(omega)
    delta = (1+alpha) * (y1-y0) / h
    a = y0
    b = y1
    c = ((y0p * alpha + y1p - delta) * h) / (1-(alpha*alpha))
    d = ((delta - y1p * alpha - y0p ) * h) / (1 - (alpha*alpha))
    
    y = a * B0(u) + b * B1(u) + c * B2(u,omega) + d * B3(u,omega)
    return y

        
def PolygonAcquisition(n,color1,color2) :
    """ Mouse acquisition of a polygon
        right click to stop
    """
    x = []  # x is an empty list
    y = []
    coord = 0
    i=0
    while n!=i:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            plt.plot(xx,yy,color1,markersize=8)
            x.append(xx)
            y.append(yy)
            plt.draw()
            #condition pour avoir seulement un label 
            if len(x) == 2 :
                plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2,label='fonction initial')
            if len(x) > 2:
                plt.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
            coord = 0
            i=i+1
    return x,y

   
def InterpolationSplineNonUniforme(n,a,b,xi,yi,sigmai,ss,color):
    """ Interpole avec une spline exponentielle des données non uniformément réparties (mais triées)"""
    # Création de hi, alpha, beta, mi, ni et deltai
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
    alpha = []
    beta = []
    for i in range(n-1):
        alpha.append(alphai(sigmai[i]*hi[i]))
        beta.append(betai(sigmai[i]*hi[i]))
        
    mi=[]
    for i in range(1,n-1):#i va de 2 à N-1
        mi.append((1-alpha[i]**2)*hi[i]*beta[i-1])  
    # on a donc mi[0] = m2 et mi[N-3] = mn-1
    ni=[]
    for i in range(1,n-1): #i va de 2 à N-1
        ni.append(((1-alpha[i-1]**2)*hi[i-1]*beta[i]))   
    # on a donc ni[0] = n2, et mi[N-3] = nn-1
    deltai=[]
    for i in range(1,n-1): #idem que pour mi et ni
        deltai.append((1+alpha[i]) * ((yi[i+1]-yi[i]) / hi[i]))
        
        
        
    # DEFINITION DE M
    M = np.zeros((n,n))
    for i in range(1,n-1) :#De i = 2 à N-1, lignes "classiques", mais indexé en 0 
        M[i][i] = alpha[i-1] * mi[i-2] + alpha[i] * ni[i-2]
        M[i][i-1] = mi[i-2]
        M[i][i+1] = ni[i-2]
    #Première et dernière lignes
    M[0][0] = alpha[0]
    M[0][1] = 1
    M[n-1][n-2] = 1
    M[n-1][n-1] = alpha[1]
    

    #DEFINIR res comme la matrice b de notre équation de la question 8
    res =np.zeros((n,1))
    for i in range(1,n-2):
        res[i][0]=deltai[i-1]*mi[i-2]+deltai[i-2]*ni[i-2]
    res[0][0] = deltai[0]
    res[n-1][0] = deltai[-1]
    
    derivees = np.linalg.solve(M,res) 
    for i in range(0,n-1):
        InterpoleB1(xi[i],yi[i],derivees[i][0],xi[i+1],yi[i+1],derivees[i+1][0],sigmai[i],ss,color)
        

 

def InterpolationSplineNonUniformeNp(n,a,b,xi,yi,sigmai,ss):
    #plt.figure()
    #plt.gca().set_xlim(a, b)
    #plt.gca().set_ylim(a, b)
    #xi,yi = PolygonAcquisition(n,'oc','c--')
    #tri(xi,yi) #On ne veut pas faire de paramétrique donc on veut que ce soit trié   
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
    alpha = []
    beta = []
    for i in range(n-1):
        alpha.append(alphai(sigmai[i]*hi[i]))
        beta.append(betai(sigmai[i]*hi[i]))
        
    mi=[]
    for i in range(1,n-1):#i va de 2 à N-1
        mi.append((1-alpha[i]**2)*hi[i]*beta[i-1])  
    # on a donc mi[0] = m2 et mi[N-3] = mn-1
    ni=[]
    for i in range(1,n-1): #i va de 2 à N-1
        ni.append(((1-alpha[i-1]**2)*hi[i-1]*beta[i]))   
    # on a donc ni[0] = n2, et mi[N-3] = nn-1
    deltai=[]
    for i in range(1,n-1): #idem que pour mi et ni
        deltai.append((1+alpha[i]) * ((yi[i+1]-yi[i]) / hi[i]))
        
    # DEFINITION DE M
    M = np.zeros((n,n))
    for i in range(1,n-1) :#De i = 2 à N-1, lignes "classiques", mais indexé en 0 
        M[i][i] = alpha[i-1] * mi[i-2] + alpha[i] * ni[i-2]
        M[i][i-1] = mi[i-2]
        M[i][i+1] = ni[i-2]

    #Première et dernière lignes
    M[0][0] = alpha[0]
    M[0][1] = 1
    M[n-1][n-2] = 1
    M[n-1][n-1] = alpha[1]
    
        

    #DEFINIR res comme la matrice b de notre équation de la question 8
    res =np.zeros((n,1))
    for i in range(1,n-2):
        res[i][0]=deltai[i-1]*mi[i-2]+deltai[i-2]*ni[i-2]
    res[0][0] = deltai[0]
    res[n-1][0] = deltai[-1]
    
    
    derivees = np.linalg.solve(M,res) 
    yfinal = []
    for i in range(0,n-1):
        y = InterpoleB1Np(xi[i],yi[i],derivees[i][0],xi[i+1],yi[i+1],derivees[i+1][0],sigmai[i],ss)
        for elem in y :
            yfinal.append(elem)
    return yfinal

 


def InterpolationParametrique(n,a,b,sigmai,ss):
    plt.figure()
    plt.gca().set_xlim(a, b)
    plt.gca().set_ylim(a, b)
    xi,yi = PolygonAcquisition(n,'oc','c--')
    
    ti = np.linspace(0, 1, len(xi)) #car a = 0 et b = 1
    
    di = [np.sqrt((yi[i+1]-yi[i])**2+(xi[i+1]-xi[i])**2) for i in range(len(xi)-1)]
    di = [(di[i]-a)/(b-a) for i in range(len(xi)-1)]
    dt = np.sum(di)
    ti = [0]
    for i in range(len(di)):
        ti.append(ti[i]+di[i])
    
    tx = InterpolationSplineNonUniformeNp(n,a,b,ti,xi,sigmai,ss)
    ty = InterpolationSplineNonUniformeNp(n,a,b,ti,yi,sigmai,ss)
    
    plt.plot(tx, ty,label='chordal parameterization')
    plt.legend()

          

###############---SCRIPT_PRINCIPALE---#####################

#########################   
##### CAS EXPLICITE #####
#########################
xi=[0,1,2,3,4,5,6,7,8,9,10]
yi=[1,2,8,8,4,4,4,4,4,2,2] 
   
plt.plot(xi,yi,"o--") # Données initiales
InterpolationSplineNonUniforme(11,0,10,xi,yi,[1,1,1,1,1,1,1,1,1,1],0.01,"b")
InterpolationSplineNonUniforme(11,0,10,xi,yi,[5,5,5,5,5,5,5,5,5,5],0.01,"g")
InterpolationSplineNonUniforme(11,0,10,xi,yi,[20,20,20,20,20,20,20,20,20,20],0.01,"k")
# La courbe bleue est équivalente à la spline naturelle
# La courbe verte a une tension globale de 5
# La courbe noire a une tension globale de 20


############################    
##### CAS PARAMETRIQUE #####
############################
#Ce cas ne fonctionne pas : les raccords ne sont pas C2(ni C1), mais sont justes continus
nombre_clic=5

a=0
b=1
ss=0.01
sigmai = [1,1,1,1,1,1,1,1,1]
InterpolationParametrique(nombre_clic,-10,10,sigmai,0.1)
