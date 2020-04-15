# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:09:52 2020

@author: Amelys Rodet
"""

#Outils
import load_tests as ldt
import plotingv2 as plot

#Fonctionnalites
import splines_naturelles as splnat
import splines_de_lissage as spllis
import Tache_4_Detection_donnes_aberrantes as ptsabe
import RANSAC as rs

if __name__ == '__main__':
    
    #Dictionnaire de choix de la fonctionnalite
    D = {'1' : ("Création d'une Spline naturelle",splnat.creation_spline_naturelle),
         '2' : ("Création d'une Spline de lissage",spllis.creation_spline_lissage),
         '3' : ('Gestionnaire de points aberrants avant création des splines',ptsabe.trouve_points_aberrants),
         '4' : ("Algorithme de RanSac",rs.Lancer_Ransac)}
    
    #Menu principal
    ldt.affiche_separation()
    print("\nBienvenue dans l'application d'interpolation Interpolaspline !\n")
    
    
    while True :
        
        print("\n----------------- Menu Principal -----------------\n")
        
        print("Veuillez choisir la fonctionnalite desiree :\n")
        
        print("Note : Pour Quitter le programme, appuyer sur q lors d'un choix.\n")
        
        for key in D.keys():
            print(key," : ",D[key][0])
        
        fonctionnalite = ldt.input_choice(list(D.keys()))
        
        ldt.affiche_separation()
        
        Retour = D[fonctionnalite][1]()
        
        if Retour is not None:
            (X,Y,f,is_tab) = Retour
            if not is_tab :
                ldt.affiche_separation()
                print("\nPoints aberrants supprimés. Que voulez-vous faire ?")
                print("1 : Creer la spline naturelle associée")
                print('2 : Creer la spline de lissage associée')
                print('3 : Retourner au menu principal')
                keep_going = ldt.input_choice(['1','2','3'])
                if keep_going == '1':
                    splnat.creation_spline_naturelle(X,Y,f)
                elif keep_going == '2':
                    spllis.creation_spline_lissage(X,Y,f)
            else:
                ldt.affiche_separation()
                print("\nPoints aberrants supprimés. Que voulez-vous faire ?")
                print("1 : Creer les splines naturelles associées")
                print('2 : Creer les splines de lissage associées')
                print('3 : Retourner au menu principal')
                keep_going = ldt.input_choice(['1','2','3'])
                if keep_going == '1':
                    splnat.creation_spline_naturelle(X,Y,f,True)
                elif keep_going == '2':
                    spllis.creation_spline_lissage(X,Y,f,True)