# neuralNetwork.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import random

# Perceptron implementation
import util
import math
import numpy as np
PRINT = True


class NeuralNetworkClassifier:
    """
    Neural Network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "neuralnetwork"
        self.max_iterations = max_iterations
        self.weights = [[], []]
        self.lamb = 0.1
        self.alpha = 0.1

    def train( self, trainingData, trainingLabels, validationData, validationLabels):
        n = len(trainingData)
        g = np.vectorize(lambda z: 1/(1+math.exp(-z)))
        self.features = trainingData[0].keys() # could be useful later
        for layer in range(0, 2):
            self.weights[layer] = [[0] * (len(self.features) + 1)] * ((-1*layer + 1)*len(self.features) + layer*len(self.legalLabels))
            for i in range(0, (-1*layer + 1)*(len(self.features) - 1) + 1):
                for j in range(0, len(self.features) + 1):
                      self.weights[layer][i][j] = random.uniform(-1/math.sqrt(len(self.features)), 1/math.sqrt(len(self.features)))
            self.weights[layer] = np.array(self.weights[layer])
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            retrain = False
            gradients = [np.zeros_like(self.weights[0]), np.zeros_like(self.weights[1])]
            D = gradients
            for i in range(n):
                x = [1] * (len(self.features) + 1)
                a = 1
                for feature in self.features:
                    x[a] = trainingData[i][feature]
                    a += 1
                x = np.array(x)
                y = [0] * len(self.legalLabels)
                y[trainingLabels[i]] = 1
                a = [x]
                error = [[]] * 3
                for layer in range(1, 3):
                    z = np.matmul(self.weights[layer - 1], a[layer - 1])
                    a.append(g(z))
                    if layer < 2:
                        a[layer] = np.insert(a[layer], 0, 1)
                error[2] = a[2] - y
                error[1] = np.matmul(self.weights[1].transpose(), error[2])
                for layer in range(1, 3):
                    gradients[layer - 1] = gradients[layer - 1] + error[layer] * a[layer - 1].transpose()


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresWeights

