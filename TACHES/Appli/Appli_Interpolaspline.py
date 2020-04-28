# -*- coding: utf - 8 -*-
"""
Avril 2020

@author: CORBILLE Clément, DOUMBOUYA Mohamed,
         EL BOUCHOUARI Zakaria,HEDDIA Bilel, 
         PIASENTIN Béryl, RODET Amélys

"""

#Outils
import load_tests as ldt
import splines_naturelles as splnat
import splines_de_lissage as spllis
import Tache_4_Detection_donnes_aberrantes as ptsabe
import RANSAC as rs
import LOESS_robuste as loess_r
import intuitive
import warnings
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) # pour enlever les DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning) # pour enlever les DeprecationWarning
    #Dictionnaire de choix de la FONCTIONNALITE
    D = {'1': ("Création d'une Spline naturelle", splnat.creation_spline_naturelle),
         '2': ("Création d'une Spline de lissage", spllis.creation_spline_lissage),
         '3': ('Méthode de gestion des points aberrants en 3 étapes',
               ptsabe.trouve_points_aberrants),
         '4': ("Algorithme de RanSac", rs.Lancer_Ransac),
         '5': ("Méthode LOESS Robuste",loess_r.Lancer_LOESS_robuste),
         '6': ("Méthode Intuitive",intuitive.Lancer_intuitive)}
            
    #Menu principal
    ldt.affiche_separation()
    print("\nBienvenue dans l'application d'interpolation Interpolaspline !\n")


    while True:

        print("\n----------------- Menu Principal -----------------\n")

        print("Veuillez choisir la FONCTIONNALITE desiree :\n")

        print("Note : Pour Quitter le programme, entrez q lors d'un choix.\n")

        for key in D:
            print(key, " : ", D[key][0])

        FONCTIONNALITE = ldt.input_choice(list(D.keys()))

        ldt.affiche_separation()

        RETOUR = D[FONCTIONNALITE][1]()

        if RETOUR is not None:
            (X, Y, F, IS_TAB) = RETOUR
            if not IS_TAB:
                ldt.affiche_separation()
                print("\nPoints aberrants traités. Que voulez-vous faire ?")
                print("1 : Creer la spline naturelle associée")
                print('2 : Creer la spline de lissage associée')
                print('3 : Retourner au menu principal')
                KEEP_GOING = ldt.input_choice(['1', '2', '3']) 
                if KEEP_GOING == '1':
                    splnat.creation_spline_naturelle(X, Y, F)
                elif KEEP_GOING == '2':
                    spllis.creation_spline_lissage(X, Y, F)
            else:
                ldt.affiche_separation()
                print("\nPoints aberrants traités. Que voulez-vous faire ?")
                print("1 : Creer les splines naturelles associées")
                print('2 : Creer les splines de lissage associées')
                print('3 : Retourner au menu principal')
                KEEP_GOING = ldt.input_choice(['1', '2', '3']) 
                if KEEP_GOING == '1':
                    splnat.creation_spline_naturelle(X, Y, F, True)
                elif KEEP_GOING == '2':
                    spllis.creation_spline_lissage(X, Y, F, True)
            
