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


def HermiteC1(x0,y0,y0p,x1,y1,y1p,ss):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.arange(x0,x1,ss)
    t = (x-x0)/(x1-x0)
    y = y0 * H0(t) + y0p * (x1-x0) * H1(t) + y1p * (x1-x0) * H2(t) + y1 * H3(t)
    plt.plot(x,y,'c',lw=2,label='cubic interpolant')
    
def HermiteC1Np(x0,y0,y0p,x1,y1,y1p,ss):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        Input :
            x0,y0,y0p,x1,y1,y1p = Hermite data of order 1 (real values)
        Return :
            plot the cubic Hermite interpolant
    """
    x = np.arange(x0,x1,ss)
    t = (x-x0)/(x1-x0)
    y = y0 * H0(t) + y0p * (x1-x0) * H1(t) + y1p * (x1-x0) * H2(t) + y1 * H3(t)
    return y

def InterpolationSplineUniforme(a,b,n,f,ss):
    #ss = samplingstep
    plt.figure()
    h = (b-a)/(n-1)
    xi = np.arange(a,b+h/2,h)
    yi = list(map(f,xi))
    M = np.eye(n)*2 + np.diag(np.concatenate([np.zeros(1), np.repeat(2,n-2), np.zeros(1)])) + np.diag(np.repeat(1,n-1), k = 1) + np.diag(np.repeat(1,n-1), k = -1) 
    res = []
    res.append(yi[1] - yi[0])
    for i in range(2,n) : #n exclu 
        res.append(yi[i] - yi[i-2])
    res.append(yi[n-1] - yi[n-2])
    res = list(map(lambda x:3/h * x,res))
    res = np.transpose(np.matrix(res)) #On bascule d'une liste à une matrice colonne
    derivees = np.linalg.solve(M,res)
    for i in range(0,n-1):
        HermiteC1(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],ss)
    x = np.arange(xi[0],xi[n-1]+ss/2,ss)
    fx = list(map(f,x))
    plt.plot(x,fx)
    plt.plot(xi,yi,"o")

def InterpolationSplineUniformeNp(a,b,n,xi,yi,ss):
    h = (b-a)/(n-1)
    M = np.eye(n)*2 + np.diag(np.concatenate([np.zeros(1), np.repeat(2,n-2), np.zeros(1)])) + np.diag(np.repeat(1,n-1), k = 1) + np.diag(np.repeat(1,n-1), k = -1) 
    res = []
    res.append(yi[1] - yi[0])
    for i in range(2,n) : #n exclu 
        res.append(yi[i] - yi[i-2])
    res.append(yi[n-1] - yi[n-2])
    res = list(map(lambda x:3/h * x,res))
    res = np.transpose(np.matrix(res)) #On bascule d'une liste à une matrice colonne
    derivees = np.linalg.solve(M,res)
    yfinal = []
    for i in range(0,n-1):
        y = HermiteC1Np(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],ss)
        for elem in y :
            yfinal.append(elem)
    return yfinal

    

    
def InterpolationSplineNonUniforme(xi,a,b,yi,ss):
    plt.figure()
    n = len(xi)
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
    M = 2 * np.diag(np.concatenate([np.ones(1), [hi[i]+hi[i+1] for i in range(n-2)], np.ones(1)])) 
    M += np.diag(np.concatenate([np.ones(1),[hi[i] for i in range(n-2)]]),k=1)  
    M += np.diag(np.concatenate([[hi[i] for i in range(1,n-1)],np.ones(1)]),k=-1)
    res = []
    res.append(1/hi[0]*(yi[1]-yi[0]))
    for i in range(1,n-1):
        res.append(hi[i-1]/hi[i] * (yi[i+1]-yi[i]) + hi[i]/hi[i-1] *(yi[i] - yi[i-1]) )
    res.append(1/hi[n-2]*(yi[n-1]-yi[n-2]))
    res = list(map(lambda x:3*x,res))
    derivees = np.linalg.solve(M,np.transpose(np.matrix(res)))
    for i in range(0,n-1):
        HermiteC1(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],ss)
    plt.plot(xi,yi,"x")
        
    
def InterpolationSplineNonUniformeNp(a,b,n,xi,yi,ss):
    hi = []
    for i in range(n-1):
        hi.append(xi[i+1]-xi[i])
    M = 2 * np.diag(np.concatenate([np.ones(1), [hi[i]+hi[i+1] for i in range(n-2)], np.ones(1)])) 
    M += np.diag(np.concatenate([np.ones(1),[hi[i] for i in range(n-2)]]),k=1)  
    M += np.diag(np.concatenate([[hi[i] for i in range(1,n-1)],np.ones(1)]),k=-1)
    res = []
    res.append(1/hi[0]*(yi[1]-yi[0]))
    for i in range(1,n-1):
        res.append(hi[i-1]/hi[i] * (yi[i+1]-yi[i]) + hi[i]/hi[i-1] *(yi[i] - yi[i-1]) )
    res.append(1/hi[n-2]*(yi[n-1]-yi[n-2]))
    res = list(map(lambda x:3*x,res))
    derivees = np.linalg.solve(M,np.transpose(np.matrix(res)))
    yfinal = []
    for i in range(0,n-1):
        y = HermiteC1Np(xi[i],yi[i],derivees[i,0],xi[i+1],yi[i+1],derivees[i+1,0],ss)
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

def InterpolationParametrique(a,b,ss):
    plt.figure()
    #x = np.arange(a,b+ss,ss)
    #plt.plot(x,list(map(ftest,x)))
    plt.gca().set_xlim(a, b)
    plt.gca().set_ylim(a, b)
    xi,yi = PolygonAcquisition('oc','c--')
    
    ti = np.linspace(a, b, len(xi)) #car a = 0 et b = 1
    # parametric interpolation according the uniform parametrization ti
    tx = InterpolationSplineUniformeNp(a,b,len(xi),ti,xi,ss)
    ty = InterpolationSplineUniformeNp(a,b,len(xi),ti,yi,ss)

    plt.plot(tx, ty,label='uniform parameterization')
    
    di = [np.sqrt((yi[i+1]-yi[i])**2+(xi[i+1]-xi[i])**2) for i in range(len(xi)-1)]
    dt = np.sum(di)
    ti = [a]
    for i in range(len(di)):
        ti.append(ti[i]+(di[i]/dt))

    # parametric interpolation according the uniform parametrization ti
    tx = InterpolationSplineUniformeNp(a,b,len(xi),ti,xi,ss)
    ty = InterpolationSplineUniformeNp(a,b,len(xi),ti,yi,ss)

    tx = InterpolationSplineNonUniformeNp(a,b,len(xi),ti,xi,ss)
    ty = InterpolationSplineNonUniformeNp(a,b,len(xi),ti,yi,ss)
    
    
    plt.plot(tx, ty,label='chordal parameterization')
    plt.legend()

def ftest(x):
    return np.sin(x**2-2*x+1) + (np.cos(x**3 + x))**2

def evalpareil(x):
    return len(set(x)) == x #set enlève tous les doublons

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

#InterpolationSplineUniforme(-1,2,10,ftest,0.01)
pareil = True
#while (pareil) :
#    xi = sorted(a+(b-a) * np.random.rand(n-2))
#    xi.insert(0,a)
#    xi.append(b)
#    pareil = evalpareil(xi)

(uk,zk) = np.loadtxt('data.txt') # Les points sont triés
plt.plot(uk,zk,"x")
a = uk[0]
b = uk[-1]
h = uk[1] - uk[0]

n = 15
N = len(uk)
xi = np.linspace(a,b, n)
h = xi[1]-xi[0] #Taille d'un intervalle entre xi , car les xi sont répartis uniformément

#MODIFIABLES POUR LES TESTS
    
#InterpolationSplineNonUniforme(uk,a,b,zk,0.01)





InterpolationParametrique(0,1,0.01)

    
    
    
