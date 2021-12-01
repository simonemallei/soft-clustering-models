import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist

class PiRoughKMeans(BaseEstimator):
    """ An estimator using Pi (Principle of Indifference) Rough K Means
        algorithm proposed in:
	"Peters, G.: Rough clustering utilizing the principle of indifference."

    Parameters
    ----------
    n_iter : int, default = 30
        Parameter used as stop criteria in case the fit
        reaches the maximum number of iterations.
    n_clusters : int, default = 10
        Parameter used to indicate the number of centroids.
    init : {'k-means++', 'random'}, default = 'k-means++'
        Parameter used to indicate the centroid initialization method.
    threshold : float, default = 1.2
        Parameter used to know whether assign an object to the
        lower bound of the closest cluster or not.
    random_state : int, default = 42
        Parameter used as seed to generate random values.
    
    """
    def __init__(self, n_iter=30, n_clusters=10, init='k-means++', threshold=1.2, random_state=42):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.init = init
        self.threshold = threshold
        self.random_state = random_state

    def __cluster_init(self, X):
        """ Initialization of the centroids using the input dataset.

        Parameters
        ----------

        X : array-like with shape (n_samples, n_features)
            The input dataset.

        Returns
        -------
        centroids : array-like with shape (n_cluster, n_features)
            The starting centroids for the execution of K-Means-like algorithm.

        """
        rng = check_random_state(self.random_state)
        centroids = np.zeros(shape=(self.n_clusters, X.shape[1]))
        if self.init == 'k-means++':
            # applying k-means++ method to assign starting centroids
            centroids[0] = np.copy(X[rng.choice(X.shape[0], 1)])
            temp_center = np.tile(centroids[0], (X.shape[0], 1))
            curr_distances = np.linalg.norm(X - temp_center, axis = 1)
            dist = np.array([curr_distances])
        
            for i in range(1, self.n_clusters):
                # computing the minimum distance from each object to a cluster
                min_distance = np.min(dist, axis = 0)
                if np.equal(np.sum(min_distance), 0.0):
                    centroid_ind = rng.choice(X.shape[0], 1)
                else:    
                    # each object has the probability to be chosen depending
                    # on its closest cluster
                    probas = min_distance / np.sum(min_distance)
                    centroid_ind = rng.choice(X.shape[0], 1, p = probas)
                centroids[i] = np.copy(X[centroid_ind])

                # appending distance from each object to the new centroid
                temp_center = np.tile(centroids[i], (X.shape[0], 1))
                curr_distances = np.linalg.norm(X - temp_center, axis = 1)
                dist = np.append(dist, np.array([curr_distances]))
        else:    
            # taking randomly k objects as starting centroids
            centroid_ind = rng.choice(X.shape[0], self.n_clusters)
            for i in range(self.n_clusters):
                centroids[i] = np.copy(X[centroid_ind[i]])
        return centroids

    def __assign(self, X):
        """ Implementation of the assignment of X to the available clusters.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to assign.

        Returns
        -------
        y : array-like with shape (n_samples, n_cluster)
            The target values (array of n_cluster values for each sample)
            y[x][k] = 1.0 if x-th sample is assigned to k-th cluster.

        """
        y = np.zeros(shape=(X.shape[0], self.n_clusters))

        # calculating distance d(v, c) from each dataset point to each centroid
        dist = cdist(self.cluster_centers_, X)
        # computing closest centroid (c_i) from each dataset point
        closest_cluster = np.argmin(dist, axis = 0)
        min_dist = np.min(dist, axis = 0)
        
        # looking for the centroids (c_j) such that
        # d(v, c_j) / d(v, c_i) <= threshold and
        # setting y[v][c_j] = 1
        for ind in range(self.n_clusters):
            inf_arr = np.full(min_dist.shape, self.threshold + 1) 
            is_closest = np.where(closest_cluster == ind, 1, 0)
            division = np.divide(dist[ind],
                                 min_dist,
                                 out=inf_arr,
                                 where=min_dist!=0.0)
            y[:,ind] = np.where((division <= self.threshold),
                                1,
                                is_closest)

        return y

    def fit(self, X, y=None):
        """ Implementation of the fitting function.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to fit.
        y : object, default = None
            Actually not used, needed for the interface.

        Returns
        -------
        self: object
            Returns the model itself.
            
        """
        # input validation
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        
        # choosing starting centroids 
        self.cluster_centers_ = self.__cluster_init(X)
        
        # assigning objects to each cluster based on starting centroids
        y_fit = self.__assign(X)

        count_iter = 0
        stop = False
        while stop != True:
            count_iter += 1
            # saving previous centroids computed
            previous_centroids = np.copy(self.cluster_centers_)
            # computing new centroids based on last assignment
            for (center, ind) in zip(self.cluster_centers_, range(self.n_clusters)):
                center_num = np.zeros(shape=(self.n_features_in_))
                center_den = np.zeros(shape=(self.n_features_in_))
                # in_cluster[i] == 1 if i-th object is in the cluster
                in_cluster = np.where(y_fit[:,ind] > 0.0, 1, 0)
                # computing new center of the cluster
                center_num = np.sum(X *(in_cluster /
                                        np.sum(y_fit, axis=1)).reshape(X.shape[0], 1),
                                    axis=0)
                center_den = np.sum(in_cluster / np.sum(y_fit, axis=1))
                # new centroid's value
                center = center_num / center_den
                self.cluster_centers_[ind] = center
            # assigning objects to each cluster based on new centroids
            y_fit = self.__assign(X)
            # stop criteria:
            # - maximum number of iterations reached
            # - equal centroids
            if (count_iter == self.n_iter or
                np.equal(previous_centroids, self.cluster_centers_).all()):
                stop = True
        self.y_ = y_fit
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.y_
        
    def predict(self, X):
        """ Implementation of the prediction function.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to predict.
        
        Returns
        -------
        y : array-like with shape (n_samples, n_cluster)
            The predicted values (array of n_cluster values for each sample)
            y[x][k] = 1.0 if x-th sample is in the k-th cluster's upper bound.

        """
        # checking whether the model is fitted or not
        check_is_fitted(self)
        
        # input validation
        X = check_array(X)
        
        y = self.__assign(X)
        return y

    def predict_proba(self, X):
        """ Implementation of the prediction probabilities function.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to predict.
        
        Returns
        -------
        y : array-like with shape (n_samples, n_cluster)
            The predicted probabilities (array of n_cluster values for each
            sample) y[x][k] indicates the probability that x-th sample is
            actually in the k-th cluster (using principle of indifference
            from prediction).
            
        """
        y = self.predict(X)
        for curr_y in y:
            curr_y /= np.sum(curr_y)
        return y
