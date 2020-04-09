Fonctions de Tache_4_Detection_donnees_aberrantes.py

####################
# Fonctions utiles #
####################

[Ligne 20]  def moyenne(x, invisibles=None)

[Ligne 47]  def ecart_type(x, moy, invisibles=None)

[Ligne 73]  def calcul_reel(i, indices)

[Ligne 94]  def voisinsI(x, i)

[Ligne 112] def voisinsKi(x, i, k)

[Ligne 132] def KNN_inter(x, k)

[Ligne 146] def isIN(x,i)

#############################################
# Méthodes de détection de points aberrants #
#############################################

[Ligne 157] def quartile(x, coeff=0.01)

[Ligne 190] def eval_quartile(x, i, a, b)

[Ligne 208] def test_Chauvenet(x, i)

[Ligne 237] def thompson(x, i, alpha=0.001)

[Ligne 265] def grubbs(x, alpha=5 / 100)

[Ligne 323] def deviation_extreme_student(x, alpha=5 / 100, borne_max=0)

[Ligne 395] def KNN(x, j, k, m)

###############################################
# Fonctions de supression de points aberrants #
###############################################

[Ligne 436] def supprime(x, methode, sup_poids=True, poids=1 / 100,k=7,m=25)

[Ligne 507] def supprime_un(x, v_poids, i, methode, sup_poids=2, poids=1 / 100)

###################################
# Gestion des intervalles d'étude #
###################################

[Ligne 555] def pas_inter(y, epsilon=0.1)

[Ligne 585] def pas_inter_essai(y, epsilon=0.1)