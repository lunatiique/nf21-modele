from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from typing import List

# définition de la classe Eleve
@dataclass
class Eleve:
    NbJoursAbsence_Plus7Absences: int
    NbJoursAbsence_Moins7Absences: int
    MoyenneParticipationEleve_DessousMoyenne: int
    MoyenneParticipationEleve_DessusMoyenne: int
    ResponsableEleve_Mere: int
    ResponsableEleve_Pere: int
    Adresse_Rural: int
    Adresse_Urbain  : int
    NbClassesEchouees  : int
    VolontePoursuiteEtudes  : int
    TravailMere_Maison  : int
    Age  : int


# on prépare le dataset 1
# on lit les données du dataset
data1 = pd.read_csv('Dataset1.csv', decimal=',')

#on supprime les données non disponibles
data1 = data1.drop('MoyenneLeverMainEleve', axis=1)
data1 = data1.drop('MoyenneLeverMainClasse', axis=1)
data1 = data1.drop('MoyenneVisiteContenuCoursEleve', axis=1)
data1 = data1.drop('MoyenneVisiteContenuCoursClasse', axis=1)
data1 = data1.drop('MoyenneCheckAnnoncesEleve', axis=1)
data1 = data1.drop('MoyenneCheckAnnoncesClasse', axis=1)
data1 = data1.drop('MoyenneParticipationsDiscussionsEleve', axis=1)
data1 = data1.drop('MoyenneParticipationsDiscussionsClasse', axis=1)
data1 = data1.drop('NiveauEtude', axis=1)
data1 = data1.drop('NbFoisEtudiantVisiteContenuCours', axis=1)
data1 = data1.drop('NbFoisParticipationEleve', axis=1)
data1 = data1.drop('NbFoisLeverMain', axis=1)
data1 = data1.drop('NbFoisUtilisateurParticipeDiscussionsGroupe', axis=1)
data1 = data1.drop('NbFoisUtilisateurCheckAnnonces', axis=1)
data1 = data1.drop('MoyenneParticipationClasse', axis=1)
data1 = data1.drop('ParentsOntRepondusSondagesEcole', axis=1)
data1 = data1.drop('ParentsSatisfaitsEcole', axis=1)

# on enlève toutes les notes
data1 = data1.drop('SujetCours', axis=1)
data1 = data1.drop('MoyenneClasse', axis=1)
data1 = data1.drop('PositionMoyenneClasse', axis=1)
data1 = data1.drop('NiveauEleveChiffre', axis=1)
data1 = data1.drop('NiveauEleveNotes', axis=1)

data1["PositionMoyenne"] = data1["PositionMoyenne"].replace(['DessusMoyenne','DessousMoyenne'],[1,0])

#on sépare la donnée à prédire :
y1 = data1["PositionMoyenne"]
data1 = data1.drop('PositionMoyenne', axis=1)

# on gère les différentes variables catégorielles.
data1_decomposed = pd.get_dummies(data1)

# on découpe les données en données d'entrainement et de test
x1_train, x1_test, y1_train, y1_test = train_test_split(data1_decomposed, y1, test_size = 0.2,random_state = 42)

#on prépare le dataset 2

# on lit les données du dataset
data2 = pd.read_csv('Dataset2.csv', decimal=',')

# On remplace les variables booléennes par de 0/1
data2["SupportScolaire"] = data2["SupportScolaire"].replace(['Oui', 'Non'], [1,0])
data2["SupportScolaireFamilial"] = data2["SupportScolaireFamilial"].replace(['Oui', 'Non'], [1,0])
data2["CoursPayants"] = data2["CoursPayants"].replace(['Oui', 'Non'], [1,0])
data2["ActivitesExtraScolaires"] = data2["ActivitesExtraScolaires"].replace(['Oui', 'Non'], [1,0])
data2["Creche"] = data2["Creche"].replace(['Oui', 'Non'], [1,0])
data2["AccesInternet"] = data2["AccesInternet"].replace(['Oui', 'Non'], [1,0])
data2["VolontePoursuiteEtudes"] = data2["VolontePoursuiteEtudes"].replace(['Oui', 'Non'], [1,0])
data2["RelationRomantique"] = data2["RelationRomantique"].replace(['Oui', 'Non'], [1,0])
data2["PositionMoyenne"] = data2["PositionMoyenne"].replace(['DessusMoyenne','DessousMoyenne'],[1,0])

#on sépare la donnée à prédire :
y2 = data2["PositionMoyenne"]
data2 = data2.drop('PositionMoyenne', axis=1)

# on enlève toutes les notes
data2 = data2.drop('MoyenneMathematiques', axis=1)
data2 = data2.drop('MoyennePortugais', axis=1)
data2 = data2.drop('MoyenneClasse', axis=1)
data2 = data2.drop('PositionMoyenneClasse', axis=1)
data2 = data2.drop('MoyenneTotaleEleve', axis=1)

# on gère les différentes variables catégorielles.
data2_decomposed = pd.get_dummies(data2)

#on regarde la corrélation entre les attributs et la position de la note de l'élève.
dict_corr = dict()
for i in range(len(data2_decomposed.corrwith(y2))):
  dict_corr[data2_decomposed.corrwith(y2)[data2_decomposed.corrwith(y2)==data2_decomposed.corrwith(y2)[i]].index[0]] = abs(data2_decomposed.corrwith(y2)[i])

sorted_dict_corr = dict(sorted(dict_corr.items(), key=lambda item: item[1], reverse=True))

# on supprime les attributs dont l'influence est en dessous de 0.20 (on garde les 6 premiers)
for key in sorted_dict_corr:
  if sorted_dict_corr[key] < 0.20:
    data2_decomposed = data2_decomposed.drop(key, axis=1)

# on découpe les données en données d'entrainement et de test
x2_train, x2_test, y2_train, y2_test = train_test_split(data2_decomposed, y2, test_size = 0.2,random_state = 42)

# on regroupe les deux datasets

# on créé le dataframe des enregistrements pour entraîner le modèle :
x_eleves_train_list: List[Eleve] = []

for row in x1_train.itertuples():
  eleve = Eleve(row.NbJoursAbsence_Plus7Absences, row.NbJoursAbsence_Moins7Absences, 
                row.MoyenneParticipationEleve_DessousMoyenne, row.MoyenneParticipationEleve_DessusMoyenne, 
                row.ResponsableEleve_Mere, row.ResponsableEleve_Pere,
                None, None, None, None, None, None)
  x_eleves_train_list.append(eleve)

for row in x2_train.itertuples():
  eleve = Eleve(None, None, None, None, None, None,
                row.Adresse_Rural, row.Adresse_Urbain, row.NbClassesEchouees, row.VolontePoursuiteEtudes,
                row.TravailMere_Maison, row.Age)
  x_eleves_train_list.append(eleve)


x_eleves_train = pd.DataFrame(x_eleves_train_list)
y_eleves_train = y1_train.append(y2_train)

# on créé le dataframe des enregistrements pour tester le modèle
x_eleves_list: List[Eleve] = []

for row in x1_test.itertuples():
  eleve = Eleve(row.NbJoursAbsence_Plus7Absences, row.NbJoursAbsence_Moins7Absences, 
                row.MoyenneParticipationEleve_DessousMoyenne, row.MoyenneParticipationEleve_DessusMoyenne, 
                row.ResponsableEleve_Mere, row.ResponsableEleve_Pere,
                None, None, None, None, None, None)
  x_eleves_list.append(eleve)

for row in x2_test.itertuples():
  eleve = Eleve(None, None, None, None, None, None,
                row.Adresse_Rural, row.Adresse_Urbain, row.NbClassesEchouees, row.VolontePoursuiteEtudes,
                row.TravailMere_Maison, row.Age)
  x_eleves_list.append(eleve)

y_eleves_test = y1_test.append(y2_test)

x_eleves_test = pd.DataFrame(x_eleves_list)

# on créé le modèle
regressionMultiple = LinearRegression()

#ignorer les valeurs NaN
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_imputed = imputer.fit_transform(x_eleves_train)
x_train_imputed = imputer.fit_transform(x_eleves_test)

# ajustement des données d'entraînement
regressionMultiple.fit(x_imputed,y_eleves_train)

# on utilise les données test pour prédire les données restantes
y_prediction = regressionMultiple.predict(x_train_imputed)
# on arrondit au plus proche :
for i in range(len(y_prediction)):
  y_prediction[i] = int(round(y_prediction[i]))
  if(y_prediction[i]==-1):
    y_prediction[i] = 0
  elif(y_prediction[i]==2):
    y_prediction[i] = 1

# on transforme les données en liste pour pouvoir les comparer
list_y_prediction = list(y_prediction.astype(int))
list_y_eleves_test = list(y_eleves_test.values)

# le score de précision des prédictions
score = 0
for i in range(len(list_y_eleves_test)):
  if(list_y_eleves_test[i] == list_y_prediction[i]):
    score+=1

score = score/len(list_y_prediction)

def answers_student():
    correct = 0
    print('Veuillez remplir ce formulaire pour prédire la réussite de cet étudiant :\n')
    while correct != 1:
      print('Âge de l\'étudiant :')
      age = input('-> ')
      if (age.isdigit()):
        age = int(age)
        correct = 1
    while correct != 2:
      print('Responsable de l\'étudiant : 1.Père / 2.Mère')
      responsable = input('-> ')
      if (responsable == '1' or responsable == '2'):
        responsable = int(responsable)
        correct = 2
    while correct != 3:
      print('Lieu de vie de l\'étudiant : 1.Rural / 2.Urbain')
      lieuVie = input('-> ')
      if (lieuVie == '1' or lieuVie == '2'):
        lieuVie = int(lieuVie)
        correct = 3
    while correct != 4:
      print('La mère de l\'étudiant travaille : 1.Oui / 2.Non')
      travail = input('-> ')
      if (travail == '1' or travail == '2'):
        travail = int(travail)
        correct = 4
    while correct != 5:
      print('Nombre de classes échouées par l\'étudiant :')
      fail = input('-> ')
      if (fail.isdigit()):
        fail = int(fail)
        correct = 5
    while correct !=6 :
      print('Fréquence de participation de l\'étudiant : 1.Supérieure à la moyenne / 2.Inférieure à la moyenne')
      participation = input('-> ')
      if (participation == '1' or participation == '2'):
        participation = int(participation)
        correct = 6
    while correct != 7:
      print('Nombre d\'absences de l\'étudiant : 1.Supérieur à la moyenne / 2.Inférieur à la moyenne')
      absences = input('-> ')
      if (absences == '1' or absences == '2'):
        absences = int(absences)
        correct = 7
    while correct != 8:
      print('L\'étudiant a t\'il l\'intention de poursuivre ses études: 1.Oui / 2.Non')
      etudes = input('-> ')
      if (etudes == '1' or etudes == '2'):
        etudes = int(etudes)
        correct = 8
    return age, responsable, lieuVie, travail, fail, participation, absences, etudes

def create_student(age, responsable, lieuVie, travail, fail, participation, absences, etudes):
    if lieuVie==1:
      vie_rural = 1  
      vie_urbain = 0
    elif lieuVie==2:
      vie_rural = 0  
      vie_urbain = 1
    if responsable==1:
      pere = 1
      mere = 0
    elif responsable==2:
      pere = 0
      mere = 1
    if participation==1:
      participation_sup = 1
      participation_moins = 0
    elif participation==2:
      participation_sup = 0
      participation_moins = 1
    if absences==1:
      absences_sup = 1
      absences_moins = 0
    elif absences==2:
      absences_sup = 0
      absences_moins = 1
    travail = (travail+1)%2
    etudes = etudes%2
    return Eleve(absences_sup, absences_moins, participation_moins, participation_sup, mere, pere, vie_rural, vie_urbain, fail, etudes, mere, age)

def predict_success(eleve_a_predire, regressionMultiple):
    test_eleve = pd.DataFrame([eleve_a_predire])
    # on utilise le modèle pour prédire le succès de l'élève en entrée
    prediction_eleve = regressionMultiple.predict(test_eleve.values)
    # on arrondit au plus proche :
    for i in range(len(prediction_eleve)):
            prediction_eleve[i] = int(round(prediction_eleve[i]))
            if(prediction_eleve[i]==-1):
                prediction_eleve[i] = 0
            elif(prediction_eleve[i]==2):
                prediction_eleve[i] = 1
    #on affiche le résultat de la prédiction :
    prediction = int(prediction_eleve[0])
    return prediction


print('\n\nBienvenue dans notre application de prédiction de réussite scolaire !\n')
age, responsable, lieuVie, travail, fail, participation, absences, etudes = answers_student()
eleve = create_student(age, responsable, lieuVie, travail, fail, participation, absences, etudes)
prediction = predict_success(eleve, regressionMultiple)
# on affiche le résultat de prédiction pour l'élève dont on a renseigné les données
if(prediction == 0):
  print("\nL'élève a de fortes chances de ne pas réussir son année scolaire.")
elif(prediction == 1):
  print('\nL\'élève a de fortes chances de réussir son année scolaire.')
else:
  print('Erreur de prédiction.')

print('\nAttention ! Ce modèle a une précision de '+str(round(score*100,2))+'% donc il arrive que les prédictions réalisées soient fausses.')