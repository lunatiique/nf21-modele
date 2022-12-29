# Prédire la prédisposition d'un étudiant à réussir ou non.

## Modèle basé sur une régression linéaire multiple

Vous trouverez ici le script python ainsi que les 2 datasets nécessaires pour prédire le succès académique d'un étudiant.

Cette prédiction sera basé sur un modèle utilisant une régression linéaire multiple utilisant les données fournies dans les datasets.

Le formulaire se présentera de la manière suivante :

```
Bienvenue dans notre application de prédiction de réussite scolaire !

Veuillez remplir ce formulaire pour prédire la réussite de cet étudiant :

Âge de l'étudiant :
-> 17
Responsable de l'étudiant : 1.Père / 2.Mère
-> 1
Lieu de vie de l'étudiant : 1.Rural / 2.Urbain
-> 2
La mère de l'étudiant travaille : 1.Oui / 2.Non
-> 1
Nombre de classes échouées par l'étudiant :
-> 0
Fréquence de participation de l'étudiant : 1.Supérieure à la moyenne / 2.Inférieure à la moyenne
-> 1
Nombre d'absences de l'étudiant : 1.Supérieur à la moyenne / 2.Inférieur à la moyenne
-> 2
L'étudiant a t'il l'intention de poursuivre ses études: 1.Oui / 2.Non
-> 1
```

Il affichera ensuite un message, si selon les informations rentrées, l'utilisateur réussira ou non :

```
L'élève a de fortes chances de réussir son année scolaire.

Attention ! Ce modèle a une précision de 85.27% donc il arrive que les prédictions réalisées soient fausses.
```