import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist

class RoughDTRSCMeans(BaseEstimator):
    """ An estimator using Rough C Means algorithm based on DTRS proposed in:
        "Li, F., Ye, M., Chen, X.: An extension to rough c-means clustering
        based on decision-theoretic rough sets model."

    Parameters
    ----------
    n_iter : int, default = 30
        Parameter used as stop criteria in case the fit reaches the
        maximum number of iterations.
    n_clusters : int, default = 10
        Parameter used to indicate the number of centroids.
    init : {'k-means++', 'random'}, default = 'k-means++'
        Parameter used to indicate the centroid initialization method.
    threshold : float, default = 0.1
        Parameter used to know whether assign an object to the lower
        bound of the closest cluster or not.
    w_low : float, default = 0.9
        Parameter used as weight of the lower bound's points of the
        cluster to calculate new centroids.
    p : float, default = 2.0
        Parameter used to get the set of neighboring points of a data point.
    m : float, default = 2.0
        Parameter used to get the conditional probability of a data point being
        in a specific cluster.
    random_state : int, default = 42
        Parameter used as seed to generate random values.
    
    """
    def __init__(self, n_iter=30, n_clusters=10, init='k-means++', threshold=0.1, w_low=0.9, p=2.0, m=2.0, random_state=42):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.init = init
        self.threshold = threshold
        self.w_low = w_low
        self.p = p
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

    def __find_norm(self, X):
        """ Implementation of computing normalization parameters.

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples that will need to be normalized
            (i.e. computing minimum and maximum value for each feature).

        """
        self.min_norm_ = np.min(X, axis=0)
        self.max_norm_ = np.max(X, axis=0)

    def normalize(self, X):
        """

        Parameters
        ----------
        X : array-like with shape (n_samples, n_features)
            The input samples to normalize using minimum and maximum
            parameter obtained by __find_norm().

        Returns
        -------
        X_norm : array-like with shape (n_samples, n_features)
            The input samples normalized using parameter
            obtained by __find_norm().

        """
        X_norm = X - np.tile(self.min_norm_, (X.shape[0], 1))
        out_arr = np.zeros(X_norm.shape)
        divide_by = np.tile(self.max_norm_ - self.min_norm_,
                            (X_norm.shape[0], 1))
        X_norm = np.divide(X_norm, divide_by, out=out_arr, where=divide_by!=0)
        return X_norm

    
    def __assign(self, X, fitting=False):
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
        # calculating distances from each object to each centroid
        dist = cdist(self.cluster_centers_, X)

        # calculating conditional probabilities P(c_i|v)
        # of (v) begin in each cluster (c_i) in (C)
        probas = np.zeros(shape=(self.n_clusters, X.shape[0]))
        # used when dist[c_i][v] == 0.0
        zeros_arr = np.zeros(shape=(X.shape[0]))

        for ind_i in zip(range(self.n_clusters)):
            den = np.zeros(shape=(X.shape[0]))
            for ind_j in zip(range(self.n_clusters)):
                dist_rates = np.divide(dist[ind_i], dist[ind_j],
                                       out=zeros_arr, where=(dist[ind_j]!=0))
                den += (dist_rates ** (2 / (self.m-1)))
            probas[ind_i] = np.full(den.shape, 1.0)
            probas[ind_i] = np.divide(1, den, out=probas[ind_i], where=(den!=0))

        y = np.zeros(shape=(X.shape[0], self.n_clusters))
        deltas = np.min(dist, axis=0) / self.p
        risk = np.zeros(shape=(self.n_clusters, X.shape[0]))

        if fitting:
            point_distances = self.point_distances_
            neighbors_list = self.neighbors_
            # self.probas_ is used during prediction as neighbors' probabilities
            # of been assigned to a certain cluster
            self.probas_ = probas
        else:
            # computing distances between each object
            # using optimized scipy method
            point_distances = cdist(X, self.X_)
            # computing neighbors list for each object
            neighbors_list = np.argsort(point_distances, axis=1)
            point_distances = np.take_along_axis(point_distances,
                                                 neighbors_list,
                                                 axis=1)
        
    
        # finding neighboring points of curr_x (v)
        is_neighbor = np.where(point_distances <=
                               deltas.reshape(point_distances.shape[0], 1),
                               1,
                               0)
        maxim_ind = np.max(np.sum(is_neighbor, axis=1))
        # ignoring the closest point because it's the object itself
        neighbors_list = neighbors_list[:,1:int(maxim_ind+1)]
        point_distances = point_distances[:,1:int(maxim_ind+1)]
        is_neighbor = is_neighbor[:,1:int(maxim_ind+1)]
        expon = np.exp(-((point_distances ** 2) / (2 * (self.std_ ** 2))))
        expon *= is_neighbor
        # computing risk R(c_i|v) for each cluster (c_i) in (C)
        for ind_ci in range(self.n_clusters):
            expons = np.where(self.probas_[ind_ci][neighbors_list.flatten()] <=
                              1 / self.n_clusters,
                              expon.flatten(),
                              0)
            expons = expons.reshape(expon.shape)
            val_lambda = np.sum(expons, axis=1) + 1
            risk[:, :] += probas[ind_ci,:] * val_lambda
            # in order to make the code behaves like val_lambda = 0.0
            # if c_j == c_i
            risk[ind_ci, :] -= probas[ind_ci,:]
        # picking the centroid (c_i) with minimum risk R(c_i|v)
        closest_cluster = np.argmin(risk, axis=0)
        min_risk = np.min(risk, axis=0)

        # looking for the centroids (c_j) such that
        # R(c_j|v) / R(c_i|v) <= 1 + threshold
        # setting y[v][c_j] = 1
        inf_arr = np.full(min_risk.shape, self.threshold + 2) 
        for ind in range(self.n_clusters):
            is_closest = np.where(closest_cluster == ind, 1, 0)
            division = np.divide(risk[ind],
                                 min_risk,
                                 out=inf_arr,
                                 where=min_risk!=0.0)
            y[:,ind] = np.where((division <= 1 + self.threshold),
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

        self.__find_norm(X)
        X = self.normalize(X)
        
        # choosing starting centroids 
        self.cluster_centers_ = self.__cluster_init(X)

        # computing distances between each object
        # using optimized scipy method
        self.point_distances_ = cdist(X, X)
        # computing neighbors list for each object
        self.neighbors_ = np.argsort(self.point_distances_, axis=1)
        self.point_distances_ = np.take_along_axis(self.point_distances_,
                                                  self.neighbors_,
                                                  axis=1)
        # computing std value
        self.std_ = np.sqrt(np.sum(self.point_distances_ ** 2) /
                            (2 * (X.shape[0] ** 2)))
        # assigning objects to each cluster based on starting centroids
        y_fit = self.__assign(X, fitting = True)
        
        count_iter = 0
        stop = False
        w_up = 1 - self.w_low
        while stop != True:
            count_iter += 1
            # saving previous centroids computed
            previous_centroids = np.copy(self.cluster_centers_)
            # computing new centroids based on last assignment
            for (center, ind) in zip(self.cluster_centers_, range(self.n_clusters)):
                center_low = np.zeros(shape=(self.n_features_in_))
                center_up = np.zeros(shape=(self.n_features_in_))
                count_low, count_up = 0, 0
                # multiplier that verifies if the object is in that cluster
                mult_centr = np.where(y_fit[:,ind] > 0.0, 1, 0)
                # multiplier for boundary elements
                mult_up = np.where(np.sum(y_fit, axis = 1) > 1.0, 1, 0)
                # multiplier for lower bound elements
                mult_low = 1 - mult_up
                # applying mult_centr condition in mult_low and mult_down
                mult_up *= mult_centr
                mult_low *= mult_centr

                # computing how many objects are in the lower bound and
                # how many are in the boundary of the cluster
                count_up = np.sum(mult_up)
                count_low = np.sum(mult_low)

                center_up = np.sum(mult_up.reshape(X.shape[0], 1) * X,
                                   axis = 0)
                center_low = np.sum(mult_low.reshape(X.shape[0], 1) * X,
                                    axis = 0)
                # computing lower bound and boundary center's components
                if count_low != 0:
                    center_low = center_low / count_low
                if count_up != 0:
                    center_up = center_up / count_up
                
                # new centroid's value
                if count_low == 0:
                    center = center_up 
                elif count_up == 0:
                    center = center_low
                else:
                    center = center_low * self.w_low + center_up * w_up
                self.cluster_centers_[ind] = center
            # assigning objects to each cluster based on new centroids
            y_fit = self.__assign(X, fitting = True)
            # stop criteria:
            # - maximum number of iterations reached
            # - equal centroids
            if (count_iter == self.n_iter or
                np.isclose(previous_centroids, self.cluster_centers_).all()):
                stop = True
        self.X_ = X
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

        X = self.normalize(X)
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
