__authors__ = ['1679933','1689435']
__group__ = 'noneyet'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        # si train_data no és de tipus float (es comprova amb dtype), es converteix (amb astype) a tipus float64
        if train_data.dtype != float:
            train_data = train_data.astype(float)

        # train_data és una matriu de 3 dimensions, però es fa la comprovació per si de cas. Si train_data té més de
        # dues dimensions, es converteix a 2D sent la primera dimensió la mateixa. 3D: P*M*N -> 2D: P*D on D=M*N,
        # amb -1 numpy calcula automàticament la D
        if train_data.ndim > 2:
            train_data = train_data.reshape(train_data.shape[0], -1)

        self.train_data = train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        # si test_data no és de tipus float (es comprova amb dtype), es converteix (amb astype) a tipus float64
        if test_data.dtype != float:
            test_data = test_data.astype(float)

        # test_data és una matriu de 3 dimensions, però es fa la comprovació per si de cas. Si test_data té més de
        # dues dimensions, es converteix a 2D sent la primera dimensió la mateixa. 3D: P*M*N -> 2D: P*D on D=M*N,
        # amb -1 numpy calcula automàticament la D
        if test_data.ndim > 2:
            test_data = test_data.reshape(test_data.shape[0], -1)

        # calcular les distàncies entre train_data i test_data i ordenar l'array resultant per columnes per obtenir
        # els primers k elements de cada una
        distances = cdist(test_data, self.train_data).argsort(axis=1)
        distances = distances[::, :k]

        # inicialitzar self.neighbors com un array 2D amb la mateixa shape que distances, però que enlloc de les
        # distàncies als k punts més propers, conté les etiquetes d'aquests punts més propers
        self.neighbors = np.array([self.labels[x] for x in np.nditer(distances)]).reshape(distances.shape[0], k)

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
