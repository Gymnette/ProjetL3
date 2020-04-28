# -*- coding: utf - 8  -*-
"""
Avril 2020

@author: Interpolaspline

"""
def is_char_or_num(char):
    """
    est-ce un char
    """
    minu = 64 < ord(char) < 91
    maju = 96 < ord(char) < 123
    nume = 47 < ord(char) < 58
    return minu or maju or nume


def indice_fonction(string):
    """
    indices des interieurs de fonction
    """
    inds = []
    dans_fonc = 0
    compteur = []
    for i, char in enumerate(string):
        if dans_fonc:
            inds.append(i)
        if i > 0 and char == '(' and is_char_or_num(string[i-1]):
            dans_fonc += 1
            compteur.append(0)
        elif char == '(' and dans_fonc:
            compteur[-1] += 1
        elif char == ')' and dans_fonc:
            if compteur[-1] == 0:
                compteur.pop(-1)
                dans_fonc -= 1
            else:
                compteur[-1] -= 1
    return inds



def liste_indices(string, sym):
    """
    Retourne la liste des indices de sym
    """
    l_ind = []
    str_simple = False
    str_double = False
    if sym == "=":
        dans_fonc = indice_fonction(string)
    else:
        dans_fonc = []
    apres_parenthese = False
    for i, char in enumerate(string):
        dans_qqch = str_simple or str_double
        if i > 0:
            apres_parenthese = string[i-1] == '(' or string[i-1] == '['
        if char == sym and not (dans_qqch or apres_parenthese or i in dans_fonc):
            l_ind.append(i)
        elif char == '"':
            str_simple = not str_simple
        elif char == "'":
            str_double = not str_double
    return l_ind


def ajouter_espace_entre_virgules(l_cour):
    """
    Ajoute les espaces apres les virgules
    """
    l_bis = l_cour
    l_ind = liste_indices(l_bis, ',')
    for i in range(len(l_ind) - 1, - 1, - 1):
        ind = l_ind[i]
        if ind < len(l_bis)-1 and l_bis[ind + 1] != " ":
            l_bis = l_bis[:ind + 1] + ' ' + l_bis[ind + 1:]

    return l_bis

def enlever_espace_avant_deux_points(l_cour):
    """
    enlever_espace_avant_deux_points
    """
    l_bis = l_cour
    l_ind = liste_indices(l_bis, ':')
    for i in range(len(l_ind) - 1, - 1, - 1):
        ind = l_ind[i]
        if l_bis[ind - 1] == " ":
            l_bis = l_bis[:ind-1]+ l_bis[ind:]

    return l_bis


def ajouter_espace_entre_operateurs(l_cour):
    """
    Ajoute les espaces avant et apres les signes == , < , <= , >= , = ,  +, * etc
    """
    l_bis = l_cour
    symbole_un_c = ['=', '<', '>', '+', '-', '*', '/', '!']

    for sym in symbole_un_c:
        if sym in l_bis:
            l_ind = liste_indices(l_bis, sym)
            for i in range(len(l_ind) - 1, - 1, - 1):
                ind = l_ind[i]
                if ind != 0 and not(l_bis[ind-1] == ' ' or l_bis[ind-1] in symbole_un_c):
                    l_bis = l_bis[:ind] + ' ' + l_bis[ind:]
                    ind += 1
                clause = l_bis[ind + 1] == ' ' or l_bis[ind + 1] in symbole_un_c
                if ind < len(l_cour) - 1 and not clause:
                    l_bis = l_bis[:ind + 1] + ' ' + l_bis[ind + 1:]

    return "".join(l_bis)

if __name__ == "__main__":
    NOM_FICHIER = input("nom du fichier :\n> ")
    NOM_FICHIER_BIS = NOM_FICHIER.split('.')[0] + "_bis." + NOM_FICHIER.split('.')[1]

    FICHIER = open(NOM_FICHIER, 'r')
    FICHIER_BIS = open(NOM_FICHIER_BIS, 'x')

    DOC = False

    for ligne in FICHIER:
        if '"""' in ligne:
            DOC = not DOC
        ligne_bis = ligne
        if not DOC:
            ligne_bis = ajouter_espace_entre_virgules(ligne_bis)
            ligne_bis = ajouter_espace_entre_operateurs(ligne_bis)
            ligne_bis = enlever_espace_avant_deux_points(ligne_bis)
        FICHIER_BIS.write(ligne_bis)


    FICHIER.close()
    FICHIER_BIS.close()
    if __name__ == "__main__":
    print("ce programme ne se lance pas seul. Lancer Appli_Interpolaspline.")
