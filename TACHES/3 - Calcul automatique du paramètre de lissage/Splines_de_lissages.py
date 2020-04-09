

"""
L'interpolation reflète toutes les variations, y compris les valeurs aberrantes(bruit)
Une spline de lissage permet de satisfaire le compromis entre 
la présence des observations "bruyantes" et son raccord aux données présentées 

Ce qui fait qu'un lissage est considéré comme différent d'une interpolation
"""
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import signaux_splines  as sign
import matplotlib.pyplot as plt

# Les 4 polynomiales cubiques formant la base de Hermite sur [0,1]
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

    
def HermiteC1(x0,y0,y0p,x1,y1,y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)
        
        Input :
            x0,y0,y0p,x1,y1,y1p : Hermite data of order 1 (real values)
        Output :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0,x1,100)
    y = y0*H0((x-x0)/(x1-x0)) + y0p*(x1-x0)*H1((x-x0)/(x1-x0)) + y1p*(x1-x0)*H2((x-x0)/(x1-x0)) + y1*H3((x-x0)/(x1-x0))    
    return x,y


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

def H03(N,n,uk,xi,H):
    """ 
    Création de la matrice HO3
    
    Input : 
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        H : vecteur de flottants (pas entre les xi)
    Output : 
        HO3 : Matrice n,n (de flottants)
    """
    M=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                M[j][i]=H0((uk[ki]-xi[i])/H[i])
                M[j][i+1]=H3((uk[ki]-xi[i])/H[i])
                j+=1
    return M


def H12(N,n,uk,xi,H):
    """ 
    Création de la matrice H12
    
    Input : 
        N : entier(taille de l'échantillon étudié), n - entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers, h - entier(pas régulier des noeuds sur l'intervalle du lissage [a,b])
        H : vecteur de flottants (pas entre les xi)
    Output:
        H12 : Matrice n,n (de flottants)
    """
    H12=np.zeros((N,n))
    j=0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki]<=xi[i+1]:
                H12[j][i]=H[i]*H1((uk[ki]-xi[i])/H[i])
                H12[j][i+1]=H[i]*H2((uk[ki]-xi[i])/H[i])
                j+=1
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
    return H03(N,n,uk,xi,H) + (np.dot(np.dot(H12(N,n,uk,xi,H),np.linalg.inv(MatriceA(n,H))),MatriceR(n,H)))


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


def Matdiag(n):
    """ 
    Création de la matrice Matdiag pour trouver y'
    
    Intput :  
        n : entier(nombre de neouds)
    Output : 
        Matdiag : Matrice n,n de flottants
    """
    Matdiag=np.zeros((n,n))
    d=[2]
    d=np.append(d,4*np.ones(n-2))
    d=np.append(d,2)
    Matdiag=np.diag(d)+np.diag(np.ones(n-1),-1)+np.diag(np.ones(n-1),1)
    return Matdiag


"""------------------------------------------------------
MAIN PROGRAM :   
------------------------------------------------------"""
if __name__ == '__main__':


    (uk,zk) =  np.loadtxt('data.txt') # prépare les donnéeS
    zk = sign.add_bivariate_noise(zk, 3) # ajoutons un peu de bruit

    model = SimpleExpSmoothing(zk) # crée la classe
    model_fit = model.fit() # met en forme le modèle
    rho = model_fit.model.params['smoothing_level']# trouve la valeur optimale
    print(rho)
    

    N = len(uk) # taille de l'échantillon
    
    plt.plot(uk,zk,'rx',label='données') # affichage des points de l'échantillon

    
    n=15 # nombre des noeuds attendus pour la spline de lissage
    plt.title('Spline de lissage avec '+str(n)+' noeuds') # titre
    m = len(uk)
    a = uk[0] # intervalle
    b = uk[m-1] # intervalle
    
   
    #Test sur une repartition des noeuds aleatoire
    rdm = np.random.rand(n-2)
    rdm.sort()
    xi = [a]
    xi = np.append(xi,rdm*(b-a) + a)
    xi = np.append(xi,[b])
    plt.scatter(xi,[0]*n,label = 'noeuds')
    
    H = [xi[i+1]-xi[i] for i in range(len(xi)-1)] # vecteur des pas de la spline
  

    
    
    Y = Vecteur_y(uk,[zk],N,xi,n,H,rho)
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(MatriceA(n,H),(np.dot(MatriceR(n,H),Y))))
    xx=[]
    yy=[]
    for i in range(n-1):
        x,y = HermiteC1(xi[i],yi[0][i],yip[0][i],xi[i+1],yi[0][i+1],yip[0][i+1])
        xx=np.append(xx,x)
        yy=np.append(yy,y)
    plt.plot(xx,yy,lw=1,label='optimal')
plt.legend()
#plt.savefig('IMG_Tache3_nonunif.png')    