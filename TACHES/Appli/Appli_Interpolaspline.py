# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:09:52 2020

@author: Amelys Rodet
"""

#Outils
import load_tests as ldt

#Foncitonnalites
import splines_naturelles as splnat

if __name__ == '__main__':
    
    #Dictionnaire de choix de la fonctionnalite
    D = {'1' : splnat.creation_spline_naturelle}
    
    
    ldt.affiche_separation()
    print("\nBienvenue dans l'application d'interpolation Interpolaspline !")
    print("Pour commencer, veuillez choisir la fonctionnalite desiree :\n")
    
    print("\nNote : Pour Quitter le programme, appuyer sur q lors d'un choix.")
    
    print("1 - Spline naturelle")
    
    fonctionnalite = ldt.input_choice(['1'])
    
    ldt.affiche_separation()
    
    D[fonctionnalite]()
    