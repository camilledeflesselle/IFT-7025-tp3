"""
Nous définissons une classe pour l'algorithme naïf bayésien, avec les méthodes suivantes :
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
	* separateByClass : permet de créer un dictionnaire où les clés sont les classes et les
	valeurs sont les instances de chaque classe
    * meanAndStd  : calcule la moyenne et l'écart type de chaque attribut par classe
    * gaussProbability : méthode qui calcule la densité de probabilité d'une distribution normale
	* calculateClassProbabilities : calcule la probabilité de chaque classe pour une instance
"""
import numpy as np
import metrics  # évaluation des performances
import math

# BayesNaif pour le modèle bayesien naif

class BayesNaif:

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		"""
		# on sépare le jeu de données train par classe
		self.separateByClass(train, train_labels)
		self.meanAndStd()
        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		self.calculateClassProbabilities(x)
		bestClasse, bestProb = None, -1
		for classe, proba in self.probabilities.items():
			# on sélectionne la classe avec la probailité la plus grande
			if bestClasse is None or proba > bestProb:
				bestProb = proba
				bestClasse = classe
		return bestClasse

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		"""
		y_pred = np.array([self.predict(x) for x in X])
		metrics.show_metrics(y, y_pred)

	def separateByClass(self, train, train_labels):
		"""
		permet de stocker un dictionnaire où les clés sont les classes et les valeurs sont les instances de chaque classe, 
		en travaillant sur un jeu d'entraînement (méthode appelée par train)

		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		train_labels : est une matrice numpy de taille nx1
		"""
		self.separated = {}
		for i in range(len(train)):
			vector = train[i]
			classe = train_labels[i]
			if (classe not in self.separated):
				self.separated[classe] = [] # liste vide
			# instance avec étiquette classe ajoutée à la liste des instances dont le label est classe
			self.separated[classe].append(vector)

	def meanAndStd(self):
		"""
		permet de stocker un dictionnaire dont les clés sont les classes du jeu de données et 
		les valeurs une liste de tuples avec la moyenne et l'écart-type de chaque attribut par classe
		"""
		self.resume = {}
		for classe, vecteurs in self.separated.items():
			# moyenne et écart type de chaque attribut par classe
			self.resume[classe] = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*vecteurs)]

	def gaussProbability(self, x, mean, std):
		"""
		C'est la méthode qui calcule la densité de probabilité d'une distribution normale de x avec
		x : la valeur d'un attribut d'une instance
		mean : la moyenne d'un attribut d'une classe
		std : l'écart-type d'un attribut d'une classe
		"""
		exponent = math.exp(-((x-mean)**2/(2*std**2)))
		return (1/(std*math.sqrt(2*math.pi))*exponent)

	def calculateClassProbabilities(self, inputVector):
		"""
		calcule la probabilité de chaque classe pour une instance inputVector donnée en entrée 
		inputVector est de taille 1xm
		"""
		self.probabilities = {}
		for classe, classeMeanStd in self.resume.items(): # boucle sur les classes
			# initialisation de la probabilité à 1
			self.probabilities[classe] = 1
			for i in range(len(classeMeanStd)): # passage dans la boucle pour chaque attribut
				# moyenne et écart type de l'attribut pour la classe
				mean, std = classeMeanStd[i]
				# valeur de l'attribut 
				x = inputVector[i]
				# avec l'hypothèse d'indépendance, on multiplie les probabilités de tous les attributs (formule de Bayes)
				self.probabilities[classe] *= self.gaussProbability(x, mean, std)





