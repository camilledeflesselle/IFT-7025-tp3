# IFT-7025-tp3 

## Authors
- Lucas Chollet
- Camille Deflesselle

## Description des classes

### Classe KNN

Cette classe est le classifieur implémenté dans le fichier *Knn.py*. Il s'agit de la classification en utilisant la méthode des K plus proches voisins. Son initialisation prend en argument trois paramètres :
 - **repeat_kfold** qui correspond à la valeur de K maximale utilisée lors de la recherche du meilleur K
 - **L** : le nombre de sous-échantillons des données d'entraînement lors de la validation croisée
 - **k** : une valeur de k utilisée lors de la phase d'évaluation, utile si on utilise le classifieur en connaissant déjà la valeur du meilleur k

 Les méthodes de cette classe sont les suivantes :
 - **euclideanDistance** : permet de calculer la distance euclidienne entre deux instances
 - **getNeighbors** : permet de renvoyer un nombre K des classes des plus proches voisins d'une instance
 - **train** : permet d'identifier les données d'entraînement
 - **predict** : permet de prédire la classe d'une instance
 - **evaluate** : permet de stocker toutes les prédictions d'un jeu de données test et d'évaluer les performances de l'algorithme (Accuracy, Précision, Rappel, Score F1 et matrice de confusion)
 - getBestKppv : permet de faire une validation croisée sur les données d'entraînement et de calculer la meilleure valeur de K
 - plotAccuracy : permet de représenter la moyenne des exactitudes obtenues lors de la validation croisée en fonction des valeurs de K. Cette fonction nous a permis d'ajuster la valeur de repeat_kfold

 ### Classe BayesNaif

Cette classe est le classifieur implémenté dans le fichier *NaiveBayes.py*. Il s'agit de la classification bayésienne naïve. Son initialisation ne prend pas d'argument en entrée.

 Les méthodes de cette classe sont les suivantes :
 - **train** : permet d'entraîner le modèle avec les données d'entraînement
 - **predict** : permet de prédire la classe d'une instance
 - **evaluate** : permet de stocker toutes les prédictions d'un jeu de données test et d'évaluer les performances de l'algorithme (Accuracy, Précision, Rappel, Score F1 et matrice de confusion)
 - **separateByClass** : permet de créer un dictionnaire où les clés sont les classes et les valeurs sont les instances de chaque classe, en travaillant sur un jeu d'entraînement (méthode appelée par train)
 - **meanAndStd**  : calcule la moyenne et l'écart type de chaque attribut par classe
 - **gaussProbability** : méthode qui calcule la densité de probabilité d'une distribution normale
 - **calculateClassProbabilities** : calcule la probabilité de chaque classe pour une instance en faisant appel à gaussProbability sur chaque attribut

## Répartition des tâches de travail entre les membres d’équipe
Pour faciliter notre collaboration, nous avons créé un dépôt git privé, sur lequel se trouve tout notre travail.

Pour ce projet, l'un des membres de l'équipe a implémenté la classe KNN et l'autre la classe BayesNaif.
Quant aux fonctions dédiées au chargement des datasets, nous les avons écrit ensemble.

De même, nous avons implémenté une boucle d'entraînement/test (fichier *entrainer_tester.py*) en utilisant ces deux classes sur les trois jeux de données étudiés en travaillant ensemble sur le fichier. Ce fichier nous a permis de connaître les temps d'exécution des différents classifieurs (temps d'entraînement + évaluation sur les données test).

Par ailleurs, nous avions au préalable implémenté le fichier *metrics.py* qui nous permet d'afficher les différentes métriques de performances, que nous utilisons dans nos classes lors de l'évaluation.

## Explication des difficultés rencontrées dans ce travail

Globalement, pour ce travail tout s'est bien déroulé. Nos réflexions se sont essentiellement tournées vers le choix des hyperparamètres pour l'implémentation de l'algorithme KNN. Pour le premier jeu de données, iris dataset, qui ne contient que 150 instances, nous avons choisi une valeur de 5, ce qui engendre des échantillons de 30 instances. Prendre une valeur plus élevée ne nous semblait pas adaptée.

Aussi, la recherche du meilleur K pour les deux autres jeux de données utilisés prend un long temps d'exécution, ce qui nous laissait penser que notre code était incorrect. Finalement, nous comprenons que cela est normal, compte tenu du nombre d'instances et de la complexité en temps de l'algorithme. 

Finalement, nous pensons nous être bien approprié ces deux algorithmes d'apprentissage et avoir bien compris leur fonctionnement.
