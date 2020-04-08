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

if __name__ == '__main__':
    
    #Dictionnaire de choix de la fonctionnalite
    D = {'1' : ('Spline naturelle',splnat.creation_spline_naturelle),
         '2' : ('Spline de lissage',spllis.creation_spline_lissage)}
    
    #Menu principal
    ldt.affiche_separation()
    print("\nBienvenue dans l'application d'interpolation Interpolaspline !\n")
    
    
    while True :
        
        print("\n------------ Menu Principal ------------\n")
        
        print("Veuillez choisir la fonctionnalite desiree :\n")
        
        print("Note : Pour Quitter le programme, appuyer sur q lors d'un choix.\n")
        
        for key in D.keys():
            print(key," : ",D[key][0])
        
        fonctionnalite = ldt.input_choice(list(D.keys()))
        
        ldt.affiche_separation()
        
        D[fonctionnalite][1]()
    