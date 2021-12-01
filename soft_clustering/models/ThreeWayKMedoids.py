import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist
import time

class ThreeWayKMedoids(BaseEstimator):
    """ An estimator using Three-Way K-Medoids algorithm proposed in:
        "Yu, H., Zhang, H.: A three-way decision clustering approach
        high dimensional data."
        
    Parameters
    ----------
    n_iter : int, default = 30
        Parameter used as stop criteria in case the fit reaches the
        maximum number of iterations.
    n_clusters : int, default = 10
        Parameter used to indicate the number of centroids.
    init : {'k-means++', 'random'}, default = 'k-means++'
        Parameter used to indicate the centroid initialization method.
    thresh_a : float, default = 0.7
        Parameter used to know whether assign an object to the lower
        bound of the closest cluster or not.
    thresh_b : float, default = 0.85
        Parameter used to know whether assign an object to the
        boundary of the closest clusters or not.
    random_state : int, default = 42
        Parameter used as seed to generate random values.
    
    """
    def __init__(self, n_iter=30, n_clusters=10, init='k-means++', thresh_a=0.7, thresh_b=0.85, random_state=42):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.init = init
        self.thresh_a = thresh_a
        self.thresh_b = thresh_b
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
            The starting centroids for the execution of K-Medoids-like algorithm.
        centroid_indices : array-like with shape (n_cluster)
            The indices from X of data object chosen as starting centroids
            (medoids).

        """
        rng = check_random_state(self.random_state)
        centroids = np.zeros(shape=(self.n_clusters, X.shape[1]))
        centroid_indices = []
        if self.init == 'k-means++':
            # applying k-means++ method to assign starting centroids
            centroid_ind = rng.choice(X.shape[0], 1)
            centroids[0] = np.copy(X[centroid_ind])
            centroid_indices.append(centroid_ind)
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
                centroid_indices.append(centroid_ind)

                # appending distance from each object to the new centroid
                temp_center = np.tile(centroids[i], (X.shape[0], 1))
                curr_distances = np.linalg.norm(X - temp_center, axis = 1)
                dist = np.append(dist, np.array([curr_distances]))
        else:    
            # taking randomly k objects as starting centroids
            centroid_indices = rng.choice(X.shape[0], self.n_clusters)
            for i in range(self.n_clusters):
                centroids[i] = np.copy(X[centroid_indices[i]])
        return (centroids, centroid_indices)

    
    def __assign(self, X):
        """ Implementation of the assignment of X to the available clusters.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to assign.
        
        Returns
        -------
        y : array-like with shape (n_samples, n_cluster, 2)
            The target values (array of (n_cluster, 2) values for each sample)
            y[x][k][0] = 1.0 if x-th sample is in the k-th cluster's core.
            y[x][k][1] = 1.0 if x-th sample is in the k-th cluster's frontier.

        """
        y = np.zeros(shape=(X.shape[0], self.n_clusters, 2))
        # computing distance from each object to each medoid
        dist = cdist(self.cluster_centers_, X)
        # computing closest medoid from each object
        cluster_number = np.argmin(dist, axis=0)
        min_dist = np.min(dist, axis=0)
        # computing alphas and betas for each possible medoid
        alphas = np.array(self.alpha_)[cluster_number].reshape(min_dist.shape)
        betas = np.array(self.beta_)[cluster_number].reshape(min_dist.shape)
        diff_a = min_dist - alphas
        pos = np.where(diff_a <= 0.0, 1, 0)
        diff_b = min_dist - betas
        bnd = np.where(diff_b <= 0.0, 1, 0)
        # removing from BND objects the POS objects
        bnd *= (1 - pos)
        # checking whether min_dist <= alpha[cluster_number] or
        # min_dist > alpha[cluster_number] and min_dist <= beta[cluster_number]
        for ind in range(self.n_clusters):
            # min_dist <= alpha[cluster_number]
            y[:,ind,0] = np.where((pos > 0.0) & (ind == cluster_number), 1, 0)
            # min_dist > alpha[cluster_number] and
            # min_dist <= beta[cluster_number]
            # assigning the object x to each cluster c_i such that
            # d(x, c_i) <= beta[cluster_number]    
            y[:,ind,1] = np.where((pos <= 0.0) & (bnd > 0.0) &
                                  (dist[ind,:] - betas <= 0.0),
                                  1,
                                  0)
            # min_dist > beta[cluster_number]
            y[:,ind,1] = np.where((pos <= 0.0) & (bnd <= 0.0),
                                  1,
                                  y[:,ind,1])
        
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
        if X.shape[0] <= 1:
            # since it's impossible to fit the model having at most 1 element
            # (d_ave and d_min are not computable)
            return self
        
        
        # computing minimum distance and average distance from each point
        # in order to obtain alphas and betas for each possible medoid
        alpha = np.empty(shape=(X.shape[0]))
        beta = np.empty(shape=(X.shape[0]))
        all_dist = cdist(X, X)
        d_ave = np.sum(all_dist, axis = 1) / (X.shape[0] - 1)
        d_min = np.partition(all_dist, 1, axis=1)[:, 1]
        alpha = d_min + self.thresh_a * d_ave
        beta = d_min + self.thresh_b * d_ave
        
        # choosing starting centroids 
        (self.cluster_centers_, self.centr_ind_) = self.__cluster_init(X)
        self.alpha_, self.beta_ = [], []
        for ind in self.centr_ind_:
            self.alpha_.append(alpha[ind])
            self.beta_.append(beta[ind])
        
        y_fit = self.__assign(X)
        variance = 0.0
        count_iter = 0
        stop = False
        while stop != True:
            count_iter += 1
            # saving previous centroids computed
            prevariance = variance
            variance = 0.0
            # computing new centroids based on last assignment
            for (center, ind) in zip(self.cluster_centers_, range(self.n_clusters)):
                intra_obj = np.max(y_fit[:,ind,:], axis=1)
                dist = np.inner(all_dist[self.centr_ind_[ind]], intra_obj)
                # computing new variance
                variance += dist
                # choosing from intra-cluster objects of a cluster, the one
                # that has minimal sum of euclidian distance between it
                # and the other points in that cluster and defines it
                # as the new centroid (medoid) of that cluster
                indexes = np.argwhere(intra_obj > 0.0)
                final_dist = np.full(all_dist.shape[0], dist + 1)
                indexes = indexes.reshape(indexes.shape[0])
                # if an object is not assigned to that cluster, it will have
                # a distance that is already greater than the previous medoid's
                # distance (dist + 1)
                final_dist[indexes] = np.sum(all_dist[indexes] * intra_obj,
                                             axis = 1)
                ind_x = np.argmin(final_dist)
                self.cluster_centers_[ind] = X[ind_x]
                self.centr_ind_[ind] = ind_x
                self.alpha_[ind] = alpha[ind_x]
                self.beta_[ind] = beta[ind_x]
            # assigning objects to each cluster based on new centroids
            y_fit = self.__assign(X)
            # stop criteria:
            # - maximum number of iterations reached
            # - equal centroids
            if (count_iter == self.n_iter or
                np.equal(prevariance, variance)):
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
        y : array-like with shape (n_samples, n_cluster, 2)
            The target values (array of (n_cluster, 2) values for each sample)
            y[x][k][0] = 1.0 if x-th sample is in the k-th cluster's core.
            y[x][k][1] = 1.0 if x-th sample is in the k-th cluster's frontier.

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
        
        y_proba = np.sum(y, axis=2) / np.sum(y, axis=(1,2)).reshape(y.shape[0],1)

        return y_proba
        
