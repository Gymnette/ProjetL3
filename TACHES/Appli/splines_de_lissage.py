

"""
L'interpolation reflète toutes les variations, y compris les valeurs aberrantes(bruit)
Une spline de lissage permet de satisfaire le compromis entre
la présence des observations "bruyantes" et son raccord aux données présentées

Ce qui fait qu'un lissage est considéré comme différent d'une interpolation
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import load_tests as ldt
import plotingv2 as plot




# Les 4 polynomiales cubiques formant la base de Hermite sur [0, 1]
def H0(t):
    """
        # Renvoie H0 la première polynomiale cubique

        Input :
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output :
            H0 : flottant
    """
    return 1 - 3 * t ** 2 + 2 * t ** 3


def H1(t):
    """
        # Renvoie H1 la deuxième polynomiale cubique

        Input :
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output :
            H1 : flottant
    """
    return t - 2 * t ** 2 + t ** 3


def H2(t):
    """
        # Renvoie H2 la troisième polynomiale cubique

        Input :
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output :
            H2 : flottant
    """
    return - t ** 2 + t ** 3


def H3(t):
    """
        # Renvoie H3 la quatrième polynomiale cubique

        Input :
            t : flottant(transformation affine de la valeur de l'échantillon étudiée)
        Output :
            H3 : flottant
    """
    return 3 * t ** 2 - 2 * t ** 3


def HermiteC1(x0, y0, y0p, x1, y1, y1p):
    """ Cubic Hermite interpolation of order 1 over 2 points x0 < x1
        (interpolation of value + first derivative)

        Input :
            x0, y0, y0p, x1, y1, y1p : Hermite data of order 1 (real values)
        Output :
            plot the cubic Hermite interpolant
    """
    x = np.linspace(x0, x1, 100)
    y = y0 * H0((x - x0) / (x1 - x0))
    y += y0p * (x1 - x0) * H1((x - x0) / (x1 - x0))
    y += y1p * (x1 - x0) * H2((x - x0) / (x1 - x0))
    y += y1 * H3((x - x0) / (x1 - x0))
    return x, y


# Création des matrices A, R, M, N pour trouver la matrice K
def MatriceA(n, H):
    """
    Création de la matrice A,  strictement diagonalement dominante et donc inversible

    Input :
        n : entier(nombre de neouds)
        H : vecteur d'entiers (pas entre les entre xi)
    Output :
        A : Matrice n, n (de flottants)
    """
    A = np.zeros((n, n))
    d = [2]
    d = np.append(d, [2 * (H[i] + H[i + 1]) for i in range(0, n - 2)])
    d = np.append(d, 2)
    dp1 = [1] + [H[i] for i in range(0, n - 2)]
    dm1 = [H[i] for i in range(1, n - 1)] + [1]
    A = np.diag(d) + np.diag(dm1, - 1) + np.diag(dp1, 1)
    return A


def MatriceR(n, H):
    """
    Création de la matrice R

    Input :
        n : entier(nombre de neouds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        R : Matrice n, n (de flottants)
    """
    R = np.zeros((n, n))
    d = [- 1 / H[0]]
    d = np.append(d, [H[i] / H[i - 1] - H[i - 1] / H[i] for i in range(1, n - 1)])
    d = np.append(d, 1 / H[n - 2])
    dm1 = [- H[i] / H[i - 1] for i in range(1, n - 1)]
    dm1.append(- 1 / H[n - 2])
    dp1 = [1 / H[0]]
    dp1 = np.append(dp1, [H[i - 1] / H[i] for i in range(1, n - 1)])
    R = np.diag(d) + np.diag(dm1, - 1) + np.diag(dp1, 1)
    return 3.0 * R


def MatriceM(n, H):
    """
    Création de la matrice M

    Input :
        n : entier(nombre de noeuds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        M : Matrice n - 2, n (de flottants)
    """
    M = np.zeros((n - 2, n))
    for i in range(n - 2):
        M[i][i] = 1 / H[i] ** 2
        M[i][i + 1] = - (1 / H[i] ** 2 + 1 / H[i + 1] ** 2)
        M[i][i + 2] = 1 / H[i + 1] ** 2
    return 3.0 * M


def MatriceN(n, H):
    """
    Création de la matrice N

    Input :
        n : entier(nombre de noeuds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        N : Matrice n - 2, n (de flottants)
    """
    N = np.zeros((n - 2, n))
    for i in range(n - 2):
        N[i][i] = 1 / H[i]
        N[i][i + 1] = (2 / H[i] - 2 / H[i + 1])
        N[i][i + 2] = - 1 / H[i + 1]
    return N


def MatriceK(n, H):
    """
    Création de la matrice K

    Input :
        n : entier(nombre de neouds)
        H : vecteur de flottants (pas entre les xi)
    Output :
        K : Matrice n, n (de flottants)
    """
    return MatriceM(n, H) + (np.dot(np.dot(MatriceN(n, H), np.linalg.inv(MatriceA(n, H))), MatriceR(n, H)))


# Création des matrices H03, H12 pour trouver la matrice H

# Création des matrices H03, H12 pour trouver la matrice H

def Copie(c1, c2, i, j, M):
    """
    Copie les colonnes c1 et c2. Le premier élément de c1 est à l'indice i,j
    c1 et c2 ont la même longueur.
    """
    for k, e in enumerate(c1):
        M[i + k][j] = e
        M[i + k][j + 1] = c2[k]
    return M

def H03(uk, xi, H):
    """
    Création de la matrice HO3

    Input :
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        H : vecteur de flottants (pas entre les xi)
    Output :
        HO3 : Matrice n, n (de flottants)
    """
    h03 = np.zeros((len(uk), len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi) - 1): #i de 1 à N - 1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while uk[k] < xi[i + 1]:
            t = (uk[k] - xi[i]) / H[i]# t^î k
            col1.append(H0(t))
            col2.append(H3(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1), 1)
        c2 = np.array(col2).reshape(len(col2), 1)
        h03 = Copie(c1, c2, kbase, i, h03) # Colonne i - 1, ligne
    return h03

def H12(uk, xi, H):
    """
    Création de la matrice H12

    Input :
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers, h - entier(pas régulier des noeuds sur
            l'intervalle du lissage [a, b])
        H : vecteur de flottants (pas entre les xi)
    Output:
        H12 : Matrice n, n (de flottants)
    """
    h12 = np.zeros((len(uk), len(xi))) #300 lignes et 30 colonnes
    k = 0
    for i in range(len(xi) - 1): #i de 1 à N - 1
        #On étudie l'intervalle [xi,xi+1]. k va aller de k i-1 à ki
        col1 = []
        col2 = []
        kbase = k
        while uk[k] < xi[i + 1]:
            t = (uk[k] - xi[i]) / H[i]# t^î k
            col1.append(H[i] * H1(t))
            col2.append(H[i] * H2(t))
            k += 1
        c1 = np.array(col1).reshape(len(col1), 1)
        c2 = np.array(col2).reshape(len(col2), 1)
        h12 = Copie(c1, c2, kbase, i, h12) # Colonne i - 1, ligne
    return h12


def MatriceH(n, uk, xi, H):
    """
    Création de la matrice H

    Input :
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        uk : tableau de flottants(valeurs en abscisse de l'échantillon)
        xi : tableau d'entiers
        h : vecteur de flottants (pas entre les xi)
    Return :
        H : Matrice n, n (de flottants)
    """
    return H03(uk, xi, H) + (np.dot(np.dot(H12(uk, xi, H), np.linalg.inv(MatriceA(n, H))), MatriceR(n, H)))


# Création de la matrice S pour trouver la matrice W
def MatriceS(n, H):
    """
    Création de la matrice S

    Input :
        n : entier(nombre de neouds)
        h : vecteur de flottants (pas entre les xi)
    Output :
        S : Matrice n, n de flottants
    """
    S = np.zeros((n - 2, n - 2))
    d = 2 * np.array(H[0:n - 2])
    dm1 = [1 / 2 * H[i] for i in range(2, n - 1)]
    dp1 = [1 / 2 * H[i] for i in range(0, n - 3)]
    S = np.diag(d) + np.diag(dm1, - 1) + np.diag(dp1, 1)
    return 1 / 3 * S


def MatriceW(n, uk, xi, H, rho):
    """
    Création de la matrice W

    Intput :
        uk : vecteur des abcisses (flottants) de l'échantillon étudié
        N : taille de uk (entier)
        xi : noeuds de lissage (flottants)
        n : taille de xi (entier)
        H : vecteur de flottants (pas entre les xi)
        rho :flottant,  paramètre de lissage qui contrôle le compromis entre
            la fidélité des données et le caractère théorique de la fonction
    Output :
        W : matrice n, n de flottants
    """
    W1 = np.dot(np.transpose(MatriceH(n, uk, xi, H)), MatriceH(n, uk, xi, H))
    W2 = np.dot(np.dot(np.transpose(MatriceK(n, H)), MatriceS(n, H)), MatriceK(n, H))
    return W1 + (rho * W2)


def Matricew(zk, n, uk, xi, H):
    """
    Création de la matrice w

    Input :
        uk, zk : vecteurs de float de l'échantillon étudié
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        xi : noeuds de lissage (flottants)
        H : vecteur de flottants (pas entre les xi)
    Output :
        w : matrice n, n de flottants
    """
    return np.transpose(np.dot(zk, MatriceH(n, uk, xi, H)))

# Calcul du vecteur y
def Vecteur_y(uk, zk, xi, n, H, rho):
    """
    Création du vecteur y

    Intput :
        uk, zk : vecteurs de float de l'échantillon étudié
        N : entier(taille de l'échantillon étudié)
        n : entier(nombre de neouds)
        xi : noeuds de lissage (flottants)
        H : vecteur de flottants (pas entre les xi)
        rho - float,  paramètre de lissage qui contrôle le compromis entre
            la fidélité des données et le caractère théorique de la fonction
    Output : y contenant la transposée des yi
    """
    return np.linalg.solve(MatriceW(n, uk, xi, H, rho), Matricew(zk, n, uk, xi, H))


def Matdiag(n):
    """
    Création de la matrice Matdiag pour trouver y'

    Intput :
        n : entier(nombre de neouds)
    Output :
        Matdiag : Matrice n, n de flottants
    """
    d = [2]
    d = np.append(d, 4 * np.ones(n - 2))
    d = np.append(d, 2)
    matdiag = np.diag(d) + np.diag(np.ones(n - 1), - 1) + np.diag(np.ones(n - 1), 1)
    return matdiag

def Repartition_chebyshev(a, b, n):
    '''
    Renvoie un tableau de points de l'intervalle [a,b] répartis façon chebyshev
    '''
    T = []
    t1 = float((a + b)) / 2
    t2 = float((b - a)) / 2
    for i in range(n):
        T.append(t1 + t2 * (np.cos((2 * i + 1) * np.pi / (2 * n + 2))))
    T.sort()
    T[0] = a
    T[-1] = b
    return T

def Repartition_aleatoire(a, b, n):
    """
    Reparti n points entre a,b et n
    """
    rdm = np.random.rand(n - 2)
    rdm.sort()
    xi = [a]
    xi = np.append(xi, rdm * (b - a) + a)
    xi = np.append(xi, [b])
    for i in range(len(xi) - 1):
        if xi[i] == xi[i + 1]:
            if i == len(xi) - 2:
                xi[i] = xi[i + 1] + xi[i - 1] / 2
            else:
                xi[i + 1] = (xi[i] + xi[i + 2]) / 2
    return xi

def Repartition_optimale(uk):
    xi = [uk[i] for i in range(0,len(uk),10)]
    if xi[-1] != uk[-1]:
        xi.append(uk[-1])
    return xi

def presence_intervalle_vide(xi, uk):
    for i, a in enumerate(xi):
        if i != len(xi)-1:
            b = xi[i+1]
            non_vide = False
            for p in uk:
                if a <= p <= b:
                    non_vide = True
            if not non_vide:
                return True
    return False


def trouve_rho(x,y):
    """
    trouve rho ideal
    """
    if len(x) >= 20:
        fold = 20
    else:
        fold = len(x)
    
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': x},
                    cv=fold) # 20-fold cross-validation
    grid.fit(y[:, None])

    rho =  grid.best_params_['bandwidth']
    
    return rho


def test_fichier(n, uk, zk, f=None, mode=None, aff_n=None, rho=1):
    """
    un test sur des donnees
    """
    #Tri
    uk, zk = ldt.sortpoints(uk, zk)

    a = min(uk) # intervalle
    b = max(uk) # intervalle

    if mode is None:
        ldt.affiche_separation()
        print("\nEntrez le mode de traitement du fichier :")
        print("1. repartition uniforme des noeuds")
        print("2. repartition de Chebichev")
        print("3. repartition aléatoire")
        print("4. repartition optimale")
        mode = ldt.input_choice(['1', '2', '3', '4'])

    if aff_n is None:
        ldt.affiche_separation()
        print("\nAfficher les noeuds ? (y = oui, n = non)")
        aff_n = ldt.input_choice()

    plt.figure()
    plt.title('spline de lissage avec ' + str(n) + ' noeuds') # titre
    plt.plot(uk, zk, 'rx', label='données') # affichage des points de l'échantillon

    if mode == '1':
        xi = np.linspace(a, b, n)
    elif mode == '2':
        xi = Repartition_chebyshev(a, b, n)
    elif mode == '3':
        #Test sur une repartition des noeuds aleatoire
        xi = Repartition_aleatoire(a, b, n)
    else:
        xi = Repartition_optimale(uk)
        n = len(xi)

    if presence_intervalle_vide(xi, uk):
        ldt.affiche_separation()
        print("\nErreur : Un intervalle vide est detecté.")
        print("Merci de changer au moins un des paramètres suivants :")
        print(" - nombre de noeuds")
        print(" - type de répartition\n")
        ldt.affiche_separation()
        sys.exit(1)

    if aff_n == 'y':
        plt.scatter(xi, [0] * len(xi), label='noeuds')

    H = [xi[i + 1] - xi[i] for i in range(len(xi) - 1)] # vecteur des pas de la spline

    Y = Vecteur_y(uk, [zk], xi, n, H, rho)
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(MatriceA(n, H), (np.dot(MatriceR(n, H), Y))))
    xx = []
    yy = []
    for i in range(n - 1):
        x, y = HermiteC1(xi[i], yi[0][i], yip[0][i], xi[i + 1], yi[0][i + 1], yip[0][i + 1])
        xx = np.append(xx, x)
        yy = np.append(yy, y)
    plt.plot(xx, yy, lw=1, label='spline de lissage avec rho = ' + str(rho))
    plt.legend()
    plt.show()

    if f is not None:
        xi = np.linspace(0, 1, 100)
        plot.plot1d1d(xi, f(xi), new_fig=False, c='g')

    ldt.affiche_separation()
    print("Spline créée !")
    ldt.affiche_separation()
    return xx, yy

def choisir_rho(uk,zk, rho_auto='y'):
    """
    Demande à l'utilisateur de choisir un rho'
    rho est le paramètre de lissage qui contrôle le compromis entre
    la fidélité des données et le caractère théorique de la fonction

    """
    if rho_auto == 'y':
        rho = trouve_rho(uk,zk)
    else:
        ldt.affiche_separation()
        print("\nChoisissez un paramètre de lissage :")
        rho = - 1
        while rho < 0:
            try:
                rho = int(input("> "))
                if rho < 0:
                    rho = - 1
                    print("Merci d'entrer un nombre valide")
            except ValueError:
                print("Merci d'entrer un nombre valide")
                rho = - 1
    return rho


def choisir_n():
    """
    Demande à l'utilisateur de choisir un n'
    """
    ldt.affiche_separation()
    print("\nChoisissez un nombre de noeuds :")
    n = - 1
    while n < 0:
        try:
            n = int(input("> "))
            if n < 5:
                n = - 1
                print("Merci d'entrer un nombre valide")
        except ValueError:
            print("Merci d'entrer un nombre valide")
            n = - 1
    return n

def creation_spline_lissage(x=None, y=None, f=None, is_array=False):
    """
    Parameters
    ----------
    x : array, optional
        The default is None.
    y : array, optional
        The default is None.
    f : function, optional
        DESCRIPTION. The default is None.
    is_array : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    print("\nCreation de la spline de lissage interpolant les donnees.\n")

    D_meth = {'1': "repartition uniforme des noeuds",
              '2': "repartition de Chebichev",
              '3': "repartition aléatoire",
              '4': "repartition optimale"}
    M = None

    if (x is None) or (y is None):
        x, y, f, M, is_array, seed = ldt.charge_donnees(D_meth)
        if seed is not None:
            print("Graine pour la génération du signal : ", seed)
    elif is_array:
        M = ldt.charge_methodes(D_meth)

    if is_array:

        ldt.affiche_separation()
        print("\nDéfinir un paramètre de lissage pour tous les fichiers ? (y = oui, n = non)")
        print("Si oui, et que vous choisissez ensuite le paramètre automatique,")
        print("alors ce paramètre sera recalculé pour chaque fichier.")
        rho_fixe = ldt.input_choice()

        if rho_fixe == 'y':
            ldt.affiche_separation()
            print("\nChoix automatique du paramètre de lissage ? (y = oui, n = non)")
            rho_auto = ldt.input_choice()
            if rho_auto == 'n':
                rho = choisir_rho([],[], 'n')

        if M == '4':
            n_fixe = 'y'
            n = 0
            ldt.affiche_separation()
            print("\nAfficher les noeuds ? (y = oui, n = non)")
            aff_n = ldt.input_choice()
        else:
            ldt.affiche_separation()
            print("\nDéfinir un nombre de noeuds constant pour tous les fichiers ? (y = oui, n = non)")
            n_fixe = ldt.input_choice()

            if n_fixe == 'y':
                n = choisir_n()
                ldt.affiche_separation()
                print("\nAfficher les noeuds ? (y = oui, n = non)")
                aff_n = ldt.input_choice()

        for i, e in enumerate(x):
            print("Fichier ", i + 1, "\nNombre de points : ", len(e))
            if rho_fixe == 'n':
                rho = trouve_rho(x[i],y[i])
                print("\nLe paramètre de lissage automatique serait : ", rho)
                print("Choisir ce paramètre de lissage ? (y = oui, n = non)")
                rho_auto = ldt.input_choice()
                rho = choisir_rho(x[i],y[i], rho_auto)
            else:
                if rho_auto == 'y':
                    rho = choisir_rho(x[i],y[i])
            if n_fixe == 'n':
                n = choisir_n()
                print("\nAfficher les noeuds ? (y = oui, n = non)")
                aff_n = ldt.input_choice()
            test_fichier(n, e, y[i], f, M, aff_n, rho)
    else:

        ldt.affiche_separation()
        rho = trouve_rho(x,y)
        print("\nLe paramètre de lissage automatique serait : ", rho)
        print("Choisir ce paramètre de lissage ? (y = oui, n = non)")
        rho_auto = ldt.input_choice()
        rho = choisir_rho(x,y, rho_auto)
        ldt.affiche_separation()
        print("\nChoisissez le mode de traitement des données :")
        for key in D_meth.keys():
            print(key, " : ", D_meth[key])

        M = ldt.input_choice(list(D_meth.keys()))
        if M == '4':
            n = 0
        else:
            n = choisir_n()
        test_fichier(n, x, y, f, M, rho=rho)

    print("Retour au menu principal...")
    ldt.affiche_separation()

if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")
