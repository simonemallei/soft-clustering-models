import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist

class ThreeWayCMeans(BaseEstimator):
    """ An estimator using Three-Way C-Means algorithm proposed in:
        "Zhang, K.: A three-way c-means algorithm."

    Parameters
    ----------
    n_iter : int, default = 30
        Parameter used as stop criteria in case the fit reaches the
        maximum number of iterations.
    n_clusters : int, default = 10
        Parameter used to indicate the number of centroids.
    init : {'k-means++', 'random'}, default = 'k-means++'
        Parameter used to indicate the centroid initialization method.
    threshold : float, default = 0.9
        Parameter used to know whether assign an object to the lower
        bound of the closest cluster or not.
    m : float, default = 2.0
        Parameter used as fuzzifier to compute membership degrees.
    random_state : int, default = 42
        Parameter used as seed to generate random values.
    
    """
    def __init__(self, n_iter=30, n_clusters=10, init='k-means++', threshold=0.9, m=2.0, random_state=42):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.init = init
        self.threshold = threshold
        self.m = m
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

    
    def __assign(self, X, mu):
        """ Implementation of the assignment of X to the available clusters.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to assign.
        mu : array-like with shape(n_samples, n_cluster)
            mu[x][k] containts the membership degree of x-th object to
            k-th cluster. 
        
        Returns
        -------
        y : array-like with shape (n_samples, n_cluster, 2)
            The target values (array of (n_cluster, 2) values for each sample)
            y[x][k][0] = 1.0 if x-th sample is in the k-th cluster's core.
            y[x][k][1] = 1.0 if x-th sample is in the k-th cluster's frontier.

        """
        # contains the maximum (f_j) of relationship vector of each object
        maxim_vec = np.max(mu, axis=1).reshape(X.shape[0], 1)
        # computing ratio membership (g_ij) for each object to each cluster
        ratio_membership = mu / maxim_vec
        # computing belonging matrix (l_ij)
        belongs = np.where(ratio_membership > self.threshold, 1.0, 0.0)
        # performing evaluation function (v_ij) for each object to each cluster
        sum_belongs = np.sum(belongs, axis=1).reshape(X.shape[0], 1)
        evaluation = ratio_membership / sum_belongs
        alpha_j = 1.0
        beta_j = np.full((X.shape[0], self.n_clusters), self.threshold)
        beta_j /= sum_belongs 
        difference_a = evaluation - alpha_j
        difference_b = evaluation - beta_j
        # if the difference between the evaluation value and the corresponding
        # beta is positive, then the object is considered in the cluster
        pos = np.where(difference_a >= 0.0, 1.0, 0.0)
        bnd = np.where(difference_b > 0.0, 1.0, 0.0)
        bnd *= (1 - pos)
        y = np.empty(shape=(X.shape[0], self.n_clusters, 2))
        y[:,:,0] = np.copy(pos)
        y[:,:,1] = np.copy(bnd)
        return y

    def __mu(self, X):
        """ Implementation of computing membership degrees for each
            object to each cluster.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            Input dataset to assign membership degree.
        
        Returns
        -------
        mu : array-like with shape(n_samples, n_cluster)
            mu[x][k] containts the membership degree of x-th object to
            k-th cluster.    

        """
        # calculating distance from each object to each cluster
        dist = cdist(self.cluster_centers_, X)
        
        # computing mu values as proposed in the paper
        mu = []
        for (ind, center) in zip(range(self.n_clusters), self.cluster_centers_):
            # dist_i contains the distance from each object to (center)
            dist_i = dist[ind].reshape(X.shape[0], 1)
            # corrected distances removing 0 distances
            corr_dist = np.where(np.equal(dist, 0.0),
                                 0.1 ** 10,
                                 dist).T
            # in order to prevent bugs (using dist instead of corr_dist was
            # still dividing by 0 sometimes)
            membership = np.where(np.equal(dist_i, 0.0),
                                  1,
                                  (dist_i / corr_dist) ** (2 / (self.m-1)))
            mu.append(1 / np.sum(membership, axis=1))
        mu = np.array(mu)
        # if distance from the object to the centroid is == 0.0, it must have
        # 1 of membership
        mu = np.where(np.equal(dist, 0.0),
                      1,
                      mu)
        return mu.T

    def __weights(self, X, y, mu):
        """ Implementation of computing weights for each object to each cluster.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            Input dataset to assign weights.
        y : array-like with shape (n_samples, n_cluster, 2)
            The target values (array of (n_cluster, 2) values for each sample)
            y[x][k][0] = 1.0 if x-th sample is in the k-th cluster's core.
            y[x][k][1] = 1.0 if x-th sample is in the k-th cluster's frontier.
        mu : array-like with shape(n_samples, n_cluster)
            mu[x][k] containts the membership degree of x-th object to
            k-th cluster.    
        
        Returns
        -------
        weights: array-like with shape(n_samples, n_cluster)
            weights[x][k] contains the weight of x-th object being assigned
            to k-th cluster.
        """   
        # verifying removing mu that are actually not used (if the cluster
        # c_i is not in B_xj)
        using_mu = mu * np.sum(y, axis=2)
        # computing weights denominators and replicating them (n_cluster) times
        weights_den = np.sum(using_mu, axis = 1).reshape(X.shape[0], 1)
        # computing actual weights
        weights = using_mu / weights_den
        return weights

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
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        stop = False
        
        # choosing starting centroids 
        self.cluster_centers_ = self.__cluster_init(X)

        # assigning objects to each cluster based on starting centroids
        mu = self.__mu(X)
        y_fit = self.__assign(X, mu)
        weights = self.__weights(X, y_fit, mu)
        count_iter = 0
        while stop != True:
            count_iter += 1
            # saving previous centroids computed
            previous_centroids = np.copy(self.cluster_centers_)
            # computing new centroids based on last assignment
            for (center, ind) in zip(self.cluster_centers_, range(self.n_clusters)):
                center_num = np.sum(X * weights[:,ind].reshape(X.shape[0], 1),
                                    axis=0)
                center_den = np.sum(weights[:,ind])
                center = center_num / center_den
                self.cluster_centers_[ind] = center
            # assigning objects to each cluster based on new centroids
            mu = self.__mu(X)
            y_fit = self.__assign(X, mu)
            weights = self.__weights(X, y_fit, mu)
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
        y : array-like with shape (n_samples, n_cluster, 2)
            The target values (array of (n_cluster, 2) values for each sample)
            y[x][k][0] = 1.0 if x-th sample is in the k-th cluster's core.
            y[x][k][1] = 1.0 if x-th sample is in the k-th cluster's frontier.

        """
        # checking whether the model is fitted or not
        check_is_fitted(self)
        
        # input validation
        X = check_array(X)
        mu = self.__mu(X)
        y = self.__assign(X, mu)
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
