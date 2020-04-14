# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import signaux_splines as ss
from random import sample
import math
import load_tests as ldt
import splines_naturelles as splnat
import splines_de_lissage as spllis

####################
# Fonctions utiles #
####################

def alea(n,nb):
    """
    Génère nb entiers aléatoires dans [0,n[
    
    Type entrées :
        n : int
        nb : int
        
    Type sorties :
        list[int] de longueur nb
    """
    return sample(list(range(n)), nb)

def trouver_ind(elem,liste):
    """
    Renvoie l'indice de l'élément de liste le plus proche de elem
    
    Type entrées :
        elem : float
        liste : list[float]
    
    Type sorties :
        int
    """
    for i in range(len(liste)):
        if liste[i] == elem :
            return i
        elif liste[i] > elem :
            if i == 0 :
                return i
            elif liste[i] - elem < elem - liste[i-1]:
                return i
            else :
                return i-1
    return len(liste)-1
        

def trouver_et_distance_ind_para(x,y,xc,yc,dist):
    """
    Renvoie la distance à la courbe de x,y, xc et yc représentant la courbe.
    
    Type entrées :
        x : float
        y : float
        xc : vecteur de float
        yc : vecteur de float
        dist : fonction : (float,float,float,float) -> float
    
    Type sorties :
        float
    """
    d = [dist(x,y,xc[i],yc[i]) for i in range(len(xc))]
    return min(d)

def calcul_erreur_courbe(xtest,ytest,xspline,yspline,dist):
    """
    Renvoie l'erreur totale des points à tester à la courbe, selon la fonction distance.
    
    Type entrées :
        xtest : vecteur de float
        ytest : vecteur de float
        xspline : vecteur de float
        yspline : vecteur de float
        dist : fonction : (float,float,float,float) -> float
        
    Type sorties :
        float
    """
    err = 0
    for i in range(len(xtest)):
        err += trouver_et_distance_ind_para(xtest[i],ytest[i],xspline,yspline,dist)
    return err

######################### 
# Fonctions de distance #
#########################
    
def d_euclidienne(x0,y0,x1,y1):
    """
    Calcul de la distance euclidienne entre (x0,y0) et (x1,y1)
    
    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float
    
    Type sorties :
        float
    """
    return math.sqrt((x0-x1)**2 + (y0 - y1)**2)
    
def d_carre(x0,y0,x1,y1):
    """
    Calcul de la distance euclidienne au carré entre (x0,y0) et (x1,y1)
    
    Type entrées :
        x0 : float
        y0 ; float
        x1 : float
        y1 : float
    
    Type sorties :
        float
    """
    return d_euclidienne(x0,y0,x1,y1)**2

def calcul_Spline_NU(X,Y,a,b,n):
    '''
    Calcul de la spline interpolant les données dans l'intervalle [a,b],
    sur n points d'interpolation non uniformes
    Si a et/ou b ne sont pas présents dans X, la spline est prolongée de manière linéaire entre a et min(x), et entre b et max(x)
    La pente vaut alors sur ces intervalles respectivement la dérivée au point min(x) et la dérivée au point max(x).
    Renvoie xres et yres, des vecteurs de réels donnant la discrétisation de la spline.
    '''
    H = [X[i+1]-X[i] for i in range(n-1)]
    #plt.scatter(X,Y,s=75,c='red',marker = 'o',label = "NU interpolation points")
    A = splnat.Matrix_NU(H)
    B = splnat.Matrix_NU_resulat(Y,H)
    Yp = np.linalg.solve(A,B)
    xres = []
    yres = []
    if X[0] != a :
        # y = cx+d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[0],Y[0])
        # On a alors y = YP[0] * x + (Y[0]-YP[0]X[0])
        xtemp = np.linspace(a,X[0],100)
        ytemp = [Yp[0] * x + (Y[0] - Yp[0]*X[0]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp
    
    for i in range(0,n-2):
        xtemp, ytemp = splnat.HermiteC1_non_affiche(X[i],Y[i],float(Yp[i]),X[i+1],Y[i+1],float(Yp[i+1]))
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        
        xres += xtemp
        yres += ytemp
    i = n-2
    xtemp, ytemp = splnat.HermiteC1_non_affiche(X[i],Y[i],float(Yp[i]),X[i+1],Y[i+1],float(Yp[i+1]))
    xtemp = list(xtemp)
    ytemp = list(ytemp)
    xres += xtemp
    yres += ytemp
    
    
    if X[-1] != b :
        # y = cx+d, avec c la pente.
        # <=> d = y - cx, en particulier en (X[-1],Y[-1])
        # On a alors y = YP[-1] * x + (Y[-1]-YP[-1]X[-1])
        xtemp = np.linspace(X[-1],b,100)
        ytemp = [Yp[-1] * x + (Y[-1] - Yp[-1]*X[-1]) for x in xtemp]
        xtemp = list(xtemp)
        ytemp = list(ytemp)
        xres += xtemp
        yres += ytemp
    
        
    return xres,yres

def calcul_Spline_lissage(uk,zk,a,b,n,rho,mode = '1') :
    """ 
    Cette fonction calcule la spline de lissage des données uk zk sur l'intervalle [a,b].
    Elle considère n noeuds, répartis selon Chevyshev.
    Le paramètre de lissage est fourni à la fonction : rho.
    Renvoie la discrétisation de la spline de lissage.
    
    Type entrées :
        uk : vecteur de float
        zk : vecteur de float
        a : float
        b : float
        n : int
        rho : float
    
    Type sorties :
        list[float],list[float]
        
    """
    N = len(uk) # taille de l'échantillon
    
    #Tri
    uk,zk = ldt.sortpoints(uk,zk)
    
    if mode is None:
        ldt.affiche_separation()
        print("\nEntrez le mode de traitement du fichier :")
        print("1. repartition uniforme des noeuds")
        print("2. repartition de Chebichev")
        print("3. repartition aléatoire")
        mode = ldt.input_choice(['1','2','3'])
    
    if mode == '1':
        xi = np.linspace(a,b,n)
    elif mode == '2':
        xi = splnat.Repartition_chebyshev(a,b,n)
    else:
        #Test sur une repartition des noeuds aleatoire
        xi = splnat.Repartition_aleatoire(a,b,n)
    
    H = [xi[i+1]-xi[i] for i in range(len(xi)-1)] # vecteur des pas de la spline
    
    Y = spllis.Vecteur_y(uk,[zk],N,xi,n,H,rho)
    yi = np.transpose(Y)
    yip = np.transpose(np.linalg.solve(spllis.MatriceA(n,H),(np.dot(spllis.MatriceR(n,H),Y))))
    xx=[]
    yy=[]
    for i in range(n-1):
        x,y = spllis.HermiteC1(xi[i],yi[0][i],yip[0][i],xi[i+1],yi[0][i+1],yip[0][i+1])
        xx=np.append(xx,x)
        yy=np.append(yy,y)
    
    return xx,yy

def calcul_Spline_para(x,y):
    """
    Calcule la spline paramétrique cubique de lissage interpolant les données.
    Renvoie sa discrétisation.
    La paramétrisation est cordale.
    
    Type des entrées :
        x : vecteur de float
        y : vecteur de float
        a : int
        b : int
    
    Type des sorties :
        (vecteur de float, vecteur de float)
    """
    a = min(x)
    b = max(x)
    n = len(x)
    T = splnat.Repartition_cordale(x,y,a,b)
    
    #Spline des (ti,xi)
    H = [T[i+1]-T[i] for i in range(n-1)]
    A = splnat.Matrix_NU(H)
    B = splnat.Matrix_NU_resulat(x,H)
    Xp = np.linalg.solve(A,B)
    Sx = []
    for i in range(0,n-1):
        _,xtemp = splnat.HermiteC1_non_affiche(T[i],x[i],float(Xp[i]),T[i+1],x[i+1],float(Xp[i+1]))
        Sx += list(xtemp)
    #Spline des (ti,yi)
    A = splnat.Matrix_NU(H)
    B = splnat.Matrix_NU_resulat(y,H)
    Yp = np.linalg.solve(A,B)
    Sy = []
    for i in range(0,n-1):
        _,ytemp = splnat.HermiteC1_non_affiche(T[i],y[i],float(Yp[i]),T[i+1],y[i+1],float(Yp[i+1]))
        Sy += list(ytemp)
    return Sx,Sy

##################################
# RANSAC : interpolation robuste #
##################################
# http://w3.mi.parisdescartes.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf 
# Nombre minimal de points ? Essai avec 3. Interpolation à chaque étape (et non approximation)
    
def ransac(x,y,nbitermax,err,dist,nbcorrect,nbpoints,rho):
    """
    Application de l'algorithme de Ransac avec un modèle de splines cubiques
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données, du nombre maximum d'itérations, 
    de l'erreur maximum acceptée entre un point et la spline, de la fonction distance associée, 
    du nombre de points nécessaires pour qu'un modèle soit considéré comme correct,
    du nombre de points pour créer la spline d'essai,
    et du paramètre de lissage
    xres et yres représentent la spline qui interpole au mieux les données d'après RANSAC

    Type entrées :
        x : vecteur de float
        y : vecteur de float
        nbitermax : int
        err : float
        dist : function : (float,float)x(float,float) -> float
        nbcorrect : int
        nbpoints : int
    Type sorties :
        xres : vecteur de float
        yres : vecteur de float
    """
    a = min(x)
    b = max(x)
    
    # Sauvegarde du meilleur modèle trouvé et de l'erreur associé
    xmod = None
    ymod = None
    errmod = -1
    
    for _ in range(nbitermax):
        # Choix d'un échantillon
        i_points = alea(len(x),nbpoints)
        x_selec = []
        y_selec = []
        for i in range(len(i_points)):
            x_selec.append(x[i_points[i]])
            y_selec.append(y[i_points[i]])
        
        # Création de la courbe à partir de l'échantillon
        x_selec,y_selec = ldt.sortpoints(x_selec,y_selec)
        xres,yres = calcul_Spline_NU(x_selec,y_selec,a,b,nbpoints)
        
        #plt.plot(x,y,"or")
        #plt.plot(x_selec,y_selec,"og")
        
        # Calcul des erreurs
        liste_inlier = list(i_points)
        for i in range(len(x)):
            if i in i_points :
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            else :
                i_associe = trouver_ind(x[i],xres)
                if dist(x[i],y[i],xres[i_associe],yres[i_associe]) <= err :                
                    liste_inlier.append(i)
                    #plt.plot(x[i],y[i],"bo")
        
        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier ! 
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            
            x_pour_spline = []
            y_pour_spline = []
            for i in range(len(liste_inlier)):
                x_pour_spline.append(x[liste_inlier[i]])
                y_pour_spline.append(y[liste_inlier[i]])
            x_pour_spline,y_pour_spline = ldt.sortpoints(x_pour_spline,y_pour_spline)
            xtemp, ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline,a,b,nbpoints,rho)
            err_temp = 0
            for i in range(len(x_pour_spline)):
                i_associe = trouver_ind(x_pour_spline[i],xtemp)
                err_temp += dist(x_pour_spline[i],y_pour_spline[i],xtemp[i_associe],ytemp[i_associe]) 
            if errmod == -1 or err_temp < errmod :
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)
    return xmod, ymod
        
def ransac_auto(x,y,err,dist,nbpoints,rho,pcorrect=0.99,para=False):
    """
    Automatisation de l'algorithme de Ransac avec un modèle de splines cubiques :
    le calcul de la proportion de points aberrants (outlier) ou non (inlier) est mise à jour au fur et à mesure.
    Une approximation aux moindres carrés est effectuée à la fin.
    On a besoin des données,
    de l'erreur maximum acceptée entre un point et la spline, de la fonction distance associée,
    du nombre de points pour créer la spline d'essai,
    du paramètre de lissage,
    et de la probabilité d'obtenir un résultat correct avec cet algorithme.
    para = True si et seulement si on souhaite une spline paramétrique.
    xres et yres représentent la spline qui interpole au mieux les données d'après RANSAC
    si exact est vraie, tous les calculs sont effectués à partir de la spline cubique d'interpolation.

    Type entrées :
        x : vecteur de float
        y : vecteur de float
        nbitermax : int
        err : float
        dist : function : (float,float)x(float,float) -> float
        nbcorrect : int
        nbpoints : int
    Type sorties :
        xres : vecteur de float
        yres : vecteur de float
    """
    a = min(x)
    b = max(x)
    
    prop_inlier = 0
    nbitermax = 500 # Sera ajusté en fonction de prop_inlier
    nbcorrect = math.floor(prop_inlier*len(x))
    
    # Sauvegarde du meilleur modèle trouvé et de l'erreur associée
    xmod = None
    ymod = None
    errmod = -1

    k = 0
    #deja_vu = []
    while k <= nbitermax :
        # Choix d'un échantillon
        i_points = alea(len(x),nbpoints)
        i_points.sort()
        # i_points contient toujours le même nombre de points distcints, il suffit de vérifier si ce sont les mêmes pour savoir si on a déjà fait ce cas.
        """fin = False
        for possibilite in deja_vu :
            fin = True
            for elem in i_points :
                if not elem in possibilite :
                    fin = False
                    break
            if fin :
                break
        if fin :
            k+=1
            continue # On a déjà étudié exactement ces indices là
        deja_vu.append(list(i_points))"""
        # JE NE SAIS PAS S'IL VAUT MIEUX TOUT CALCULER OU FAIRE UNE RECHERCHE A CHAQUE FOIS
        
        
        
        #print(i_points)
        x_selec = []
        y_selec = []
        for i in range(len(i_points)):
            x_selec.append(x[i_points[i]])
            y_selec.append(y[i_points[i]])
        
        # Création de la courbe à partir de l'échantillon
        if not para :
            x_selec,y_selec = ldt.sortpoints(x_selec,y_selec)
        xres,yres=[],[]
        if para :
            xres,yres = calcul_Spline_para(x_selec,y_selec)
            #plt.plot(x_selec,y_selec,"ro")
            #print(x_selec,y_selec)
            #plt.plot(xres,yres,"bo")
        else :
            xres,yres = calcul_Spline_NU(x_selec,y_selec,a,b,nbpoints)
        
        #plt.plot(x,y,"or")
        #plt.plot(x_selec,y_selec,"og")
        
        # Calcul des erreurs
        liste_inlier = list(i_points)
        for i in range(len(x)):
            if i in i_points :
                # Inutile de calculer dans ce cas là, déjà compté
                continue
            else :
                i_associe = -1
                d_courbe = -1
                if para :
                    d_courbe = trouver_et_distance_ind_para(x[i],y[i],xres,yres,dist)
                else :
                    i_associe = trouver_ind(x[i],xres)
                    d_courbe = dist(x[i],y[i],xres[i_associe],yres[i_associe])
                if d_courbe <= err :      
                    liste_inlier.append(i)
                #else :
                    #print(i)
                    #print(i_associe)
                    #print(dist(x[i],y[i],xres[i_associe],yres[i_associe]))
                    #print("fin")
        
        if len(liste_inlier) >= nbcorrect:
            # Le modèle semble ne contenir que des inlier ! 
            # On calcule la spline de lissage correspondante, avec l'erreur.
            # On garde la spline de lissage ayant l'erreur la plus petite.
            liste_inlier.sort()
            x_pour_spline = []
            y_pour_spline = []
            for i in range(len(liste_inlier)):
                x_pour_spline.append(x[liste_inlier[i]])
                y_pour_spline.append(y[liste_inlier[i]])
            if not para :
                x_pour_spline,y_pour_spline = ldt.sortpoints(x_pour_spline,y_pour_spline)
            xtemp,ytemp = 0,0
            
            if para :
                xtemp,ytemp = calcul_Spline_para(x_pour_spline,y_pour_spline)
            else :
                xtemp,ytemp = calcul_Spline_lissage(x_pour_spline, y_pour_spline,a,b,nbpoints,rho)
                        
            err_temp = -1
            if para :
                err_temp = calcul_erreur_courbe(x_pour_spline,y_pour_spline,xtemp,ytemp,dist)
                #print(i_points,err_temp)
            else :
                err_temp = 0
                for i in range(len(x_pour_spline)):
                    i_associe = trouver_ind(x_pour_spline[i],xtemp)
                    err_temp += dist(x_pour_spline[i],y_pour_spline[i],xtemp[i_associe],ytemp[i_associe]) 
            
            if errmod == -1 or err_temp < errmod :
                errmod = err_temp
                xmod = list(xtemp)
                ymod = list(ytemp)
                
                #ESSAI D'AFFICHAGE DES POINTS ABERRANTS
                #plt.close('all')
                plt.plot(x,y,"+b")
                plt.plot(x_pour_spline,y_pour_spline,"+y")
                
                # AFFICHAGE DE LA SPLINE INTERMEDIAIRE
                # A LAISSER. DANS L'IDEE, OPTION A PROPOSER
                # Mais attention à bien modifier la légende
                #plt.plot(xres,yres,"--g")
            
            # Que le modèle soit retenu ou non, on met à jour la proportion d'inliers et ce qui est associé 
   
            if prop_inlier < len(liste_inlier)/len(x):
                # Il y a plus d'inlier que ce qu'on pensait
                prop_inlier = len(liste_inlier)/len(x)
                if prop_inlier == 1 :
                    break; # On a trouvé un modèle correct, pour lequel il n'y aurait pas de points aberrants
                nbitermax = math.floor(np.log(1 - pcorrect)/np.log(1-prop_inlier**len(x))) # Sera ajusté en fonction de prop_inlier
                if nbitermax > 500 :
                    nbitermax = 500
                nbcorrect = math.floor(prop_inlier*len(x))
        k += 1
    
    return xmod, ymod

def lancement_ransac(x,y,err,rho,nconsidere=-1):
    x,y = ldt.sortpoints(x,y)
    if (nconsidere == -1):
        nconsidere = len(x)//2
    xres,yres = ransac_auto(x,y,err,d_euclidienne,nconsidere,rho)
    plt.plot(xres,yres,"r")        
    
def lancement_ransac_para(x,y,err,rho,nconsidere=-1):
    if (nconsidere == -1):
        nconsidere = len(x)//2
    xres,yres = ransac_auto(x,y,err,d_euclidienne,nconsidere,rho,para=True)
    plt.plot(xres,yres,"r")  
        
if __name__ == "__main__":
    plt.close('all')
    plt.figure()
    
    ##########################################
    # Tests fonctionnels (paramètres réglés) #
    ##########################################
    
    # Utilisation : mettre le numéro de l'exemple ici. 0 <= num <= 31
    num = 32
    # 11 
    # 14 : paramètres pas trouvés
    
    # Données de CAO, nombreuses, sans points aberrants
    
    if num == 0: # Données de CAO
        x,y = np.loadtxt('Tests\\data.txt')
        lancement_ransac(x,y,0.5,0.1)
        xreel,yreel = calcul_Spline_lissage(x,y,min(x),max(x),len(x),0.1)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : données de CAO")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
   
    # Petits tests spécifiques
    
    elif num == 1 : # Droite plus ou moins nulle, valeurs aberrantes
        x,y = np.loadtxt('droite_environ_nulle_aberrant.txt')
        lancement_ransac(x,y,2,5)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ nulle, données aberrantes")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 2 : # Droite plus ou moins nulle, pas de valeurs aberrantes
        x,y = np.loadtxt('droite_environ_nulle_pasaberrant.txt')
        lancement_ransac(x,y,0.2,0.1,nconsidere=8)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ nulle")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 3 : # Droite identité parfaite
        x,y = np.loadtxt('droite_identite.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite identité")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 4 : # Droite plus ou moins identité, valeurs aberrantes
        x,y = np.loadtxt('droite_identite_environ_aberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ identité, données aberrantes")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 5 : #Plus ou moins l'identité, sans valeurs aberrantes
        x,y = np.loadtxt('droite_identite_environ_pasaberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = x
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite environ identité")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 6 : # Droite nulle avec une donnée aberrante
        x,y = np.loadtxt('droite_nulle_un_aberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite nulle, donnée aberrante")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif num == 7 :# Droite nulle sans données aberrantes
        x,y = np.loadtxt('droite_nulle_pasaberrant.txt')
        lancement_ransac(x,y,0.5,1)
        xreel = x
        yreel = np.repeat(0,len(x))
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : droite nulle")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    # Signaux stationnaires
    
    elif 8 <= num and num <= 13 :
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        x,y, f,seed = ss.stationary_signal((30,), 0.9, noise_func=nfunc,seed=num-8)
        lancement_ransac(x,y,0.2,0.001)
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal stationnaire de régularité 0.9. seed = "+str(num-8))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    elif 14 <= num and num <= 19 :
        # JE N'ARRIVE PAS A TROUVER DE BONS PARAMETRES ICI
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        x,y, f,seed = ss.stationary_signal((30,), 0.5, noise_func=nfunc,seed=num-14)
        if num == 19 :
            lancement_ransac(x,y,0.05,0.00001,nconsidere=26)
        elif num == 15 :
            lancement_ransac(x,y,0.2,0.00002,nconsidere=26)
        elif num == 16 :
            lancement_ransac(x,y,0.1,0.00001,nconsidere=20)
        elif num == 18 :
            lancement_ransac(x,y,0.5,0.00001,nconsidere=20)
        else :# 14 et 17
            lancement_ransac(x,y,0.05,0.00001,nconsidere=24)
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal stationnaire de régularité 0.5. seed = "+str(num-14))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    # Signaux non stationnaires    
    elif 20 <= num <= 25 :
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        x, y, f,seed = ss.non_stationary_signal((30,), switch_prob=0.1, noise_func=nfunc,seed=num-20)
        if num == 21 :
            lancement_ransac(x,y,0.1,0.0005,nconsidere=20)
        else :
            lancement_ransac(x,y,0.2,0.00001)
            
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal non stationnaire avec une probabilité de saut de 0.1. seed = "+str(num-20))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
    
    elif 26 <= num <= 31 : 
        nfunc = lambda x: ss.add_bivariate_noise(x, 0.05, prob=0.15)
        x, y, f,seed = ss.non_stationary_signal((30,), switch_prob=0.2, noise_func=nfunc,seed=num-26)
        if num == 27:
            lancement_ransac(x,y,0.05,0.001,nconsidere=20) 
        if num == 29 or num == 30 or num == 31:
            lancement_ransac(x,y,0.1,0.00001,nconsidere=20) 
        else : 
            lancement_ransac(x,y,0.2,0.00001,nconsidere=25)   
        xreel = x
        yreel = f(x)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : Signal non stationnaire avec une probabilité de saut de 0.1. seed = "+str(num-26))
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])


    # PARAMETRIQUE
    elif num == 32 :
        x,y = ldt.load_points('Tests\\2D2.txt')
        xreel = list(x)
        xreel.pop(8)
        yreel = list(y)
        yreel.pop(8)
        lancement_ransac_para(x,y,2,1,nconsidere=len(x)//2)
        plt.plot(xreel,yreel,"--b")
        plt.title("Ransac : paramétrique")
        plt.legend(["Données aberrantes","Données non aberrantes","interpolation aux moindres carrées obtenue","interpolation attendue"])
