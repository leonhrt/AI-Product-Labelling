__authors__ = ['1679933', '1689435']
__group__ = '100'

import numpy as np
import utils
import scipy.spatial.distance as sc


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        # si X no és de tipus float, es converteix a tipus float amb astype()
        if X.dtype != float:
            X = X.astype(float)

        # si X té més de 2 dimensions es converteix a 2D amb reshape(-1, shape[-1]), on shape[-1] és la llargada de la
        # última dimensió i -1 serveix perquè numpy calculi la primera dimensió resultant corresponent
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if 'threshold' not in options:
            options['threshold'] = 20

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        # si self.options['km_init'] == first assignar a temp els primers K elements diferents de X
        if self.options['km_init'].lower() == 'first':
            # creació d'un array temp que contindrà els centroides amb els que posteriorment inicialitzarem
            # self.centroids i self.old_centroids
            temp = []
            # iterar per l'array X per obtenir els primers K valors diferents a temp[]
            for x in self.X:
                # si temp[] ja conté K valors, deixar d'iterar
                if len(temp) >= self.K:
                    break
                # comprovar si el punt de X en el que es troba la iteració està ja en l'array temp[], si no hi és,
                # s'afageix. Per la comprovació s'utilitza un bucle per comparar per cada element de l'array temp[]
                # l'element actual de X (x), any retorna True si coincideix algun dels elements de temp[] amb x
                if not any(np.array_equal(x, p) for p in temp):
                    temp.append(x)

            self.centroids = np.array(temp, dtype=float)
            self.old_centroids = np.array(temp, dtype=float)

        # si self.options['km-init'] == random assignar a self.centroids punts aleatòris no repetits de X
        elif self.options['km_init'].lower() == 'random':
            # passar els punts de self.X a un diccionari per així no tenir duplicats
            temp_dict = {tuple(point): 1 for point in self.X}
            # fer de les claus del diccionari un array
            unique_points = np.array(list(temp_dict.keys()))
            # ordenar aleatòriament l'array
            np.random.shuffle(unique_points)
            # assignar els primers self.K punts de l'array aleatòri
            self.centroids = unique_points[:self.K].astype(float)
            self.old_centroids = np.copy(self.centroids)

        elif self.options['km_init'].lower() == 'custom':
            # inicialitzar centroids amb un d'aleatòri
            temp = [self.X[np.random.choice(self.X.shape[0])]]
            # calcular la distància euclidiana al quadrat entre cada punt de X i el primer centroide
            distances = np.sum((self.X - temp[0]) ** 2, axis=1)

            # iterar per tenir K centroides
            for _ in range(1, self.K):
                # calcular probabilitat de selecció per cada punt
                probabilities = distances / np.sum(distances)
                # nou punt aleatòri basant-se en les probabilitats
                centroid = self.X[np.random.choice(self.X.shape[0], p=probabilities)]
                # afegir el punt nou a la llista
                temp.append(centroid)
                # calcular distànces euclidianes, ara per cada punt de X i el nou centroide
                temp_dist = np.sum((self.X - centroid) ** 2, axis=1)
                # actualitzcentroidar les distàncies amb les mínimes
                distances = np.minimum(distances, temp_dist)

            self.centroids = np.array(temp)
            self.old_centroids = np.copy(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        centroids_distance = distance(self.X, self.centroids)
        self.labels = np.argmin(centroids_distance, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()

        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """

        # inicialitzar centroides i reestablir num_iter en 0
        self._init_centroids()
        self.num_iter = 0

        # iterar mentres num_iter sigui més petit que max_iter
        while self.num_iter < self.options['max_iter']:
            # trobar quin és el centroide més proper per cada punt de la imatge
            self.get_labels()
            # calcular nous centroides
            self.get_centroids()
            # deixar d'iterar quan convergeix
            if self.converges():
                break
            # si no convergeix, augmentar en 1 el nombre d'iteracions per seguir iterant fins que convergeixi o
            # num_iter == max_iter
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        wcd = 0
        for i, distances in enumerate(distance(self.X, self.centroids)):
            wcd += distances[self.labels[i]] ** 2
        return wcd

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        threshold = self.options['threshold'] / 100
        self.K = 2
        self.fit()
        wcd_old = self.withinClassDistance()

        for i in range(self.K + 1, max_K):
            self.K = i
            self.fit()
            wcd_actual = self.withinClassDistance()
            if 1 - (wcd_actual / wcd_old) < threshold:
                self.K -= 1
                break
            wcd_old = wcd_actual


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return sc.cdist(X, C)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    # guardar a array_prob l'array K*11 de probabilitats de cada color per cada centroide
    array_prob = utils.get_color_prob(centroids)
    # amb np.argmax obtenim els indexs dels valors més alts de les probabilitats de colors
    # de cada centroide (axis=1 per obtenir valors màxims de files) i amb aquests indexs fem la llista d'etiquetes a
    # partir de utils.colors
    return utils.colors[np.argmax(array_prob, axis=1)]
