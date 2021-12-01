import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist

class ThreeWayCESpectral(BaseEstimator):
    """ An estimator using Three-Way CE Spectral algorithm proposed in:
        "Wang, P., Yao, Y.: Ce3: a three-way clustering method
        based on mathematical morphology."

    Parameters
    ----------
    n_iter : int, default = 30
        Parameter used as stop criteria in case the fit reaches the
        maximum number of iterations.
    n_clusters : int, default = 10
        Parameter used to indicate the number of centroids.
    init : {'k-means++', 'random'}, default = 'k-means++'
        Parameter used to indicate the centroid initialization method.
    std : float, default = 1.0
        Parameter used for spectral clustering.
    q : int, default = 15
        Parameter used to know the size of the neighborhoods to consider.
    rho : float, default = 1.5
        Parameter used as threshold in order to understand whether an object
        is in a cluster's lower bound or not.
    random_state : int, default = 42
        Parameter used as seed to generate random values.
    
    """
    def __init__(self, n_iter=30, n_clusters=10, init='k-means++', std=1.0, q=15, rho=1.5, random_state=42):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.init = init
        self.std = std
        self.q = q
        self.rho = rho
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

    

    def __assign_kmeans(self, X):
        """ Implementation of the assignment of X to the available clusters
            obtained through K-Means algorithm. 

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
        # computing distances from each object to each centroid
        X = X.real
        dist = cdist(self.cluster_centers_, X)

        # assigning each object to the closest centroid
        clusters = np.argmin(dist, axis=0)

        # assigning, for each cluster, the objects that have their centroid
        # as the closer one
        for ind in range(self.n_clusters):
            y[:,ind] = np.where(clusters == ind, 1, 0)

        return y

    def __kmeans(self, X):
        """ Implementation of K-Means algorithm.

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
        # computing starting centroids
        self.cluster_centers_ = self.__cluster_init(X)
        y = self.__assign_kmeans(X)
        
        stop = False
        count_iter = 0
        while stop != True:
            count_iter += 1
            previous_centroids = np.copy(self.cluster_centers_)
            # compute new centroids
            for ind in range(self.n_clusters):
                if np.sum(y[:, ind]) > 0.0:
                    center = np.sum(X * (y[:, ind].reshape(X.shape[0], 1)), axis=0) / np.sum(y[:, ind])
                    self.cluster_centers_[ind] = center
            # new assignment of object based on new centroids
            y = self.__assign_kmeans(X)
            if (count_iter == self.n_iter or
                np.equal(previous_centroids, self.cluster_centers_).all()):
                stop = True
            
        return y
    
    def __spectral(self, X):
        """ Implementation of Spectral Clustering algorithm.

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
        # computing for each object the distance from all the other objects
        X = X.real
        dist = cdist(X, X)
        # computing the affinity matrix W
        affinity = np.exp(-(dist / (2 * (self.std ** 2))))
        # computing diagonal matrix D
        diag = np.zeros(shape=(affinity.shape))
        np.fill_diagonal(diag, np.sum(affinity, axis=1))


        inv_sqrt = np.sqrt(diag)
        np.fill_diagonal(inv_sqrt, 1 / np.diagonal(inv_sqrt))
        # computing laplacian matrix L  
        laplacian = np.matmul(np.matmul(inv_sqrt, affinity), inv_sqrt)
        # eig_vec[:, i] contains the i-th normalized eigenvector such that
        # eig_val[i] corresponds to its eigenvalue
        eig_val, eig_vec = np.linalg.eig(laplacian)
        eig_val, eig_vec = eig_val[:self.n_clusters], eig_vec[:,:self.n_clusters]
        # the matrix containing the first k eigenvectors must be
        # 2-normalized through its rows
        norm_matr = eig_vec / (np.sqrt(np.sum(eig_vec ** 2, axis=1).reshape(eig_vec.shape[0], 1)))
        y = self.__kmeans(norm_matr)
        self.X_norm_ = norm_matr.real
        
        return y

    def __assign_two_way(self, X):
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
        if (X.shape == self.X_.shape and (X == self.X_).all()):
            return self.y_two_way_
        # using two pointers while-loop in order to have predict with
        # O(max(|X|, |self.X_|) * log(max(|X|, |self.X_|))) complexity
        names = [str(i) for i in range(self.n_features_in_)]
        dataset_X = np.empty(shape=(self.X_.shape[0]),
                             dtype=[(name, '<f8') for name in names])
        predict_X = np.empty(shape=(X.shape[0]),
                             dtype=[(name, '<f8') for name in names])
        for (curr_x, ind_x) in zip(self.X_, range(dataset_X.shape[0])):
            dataset_X[ind_x] = tuple(curr_x)
        for (curr_x, ind_x) in zip(X, range(predict_X.shape[0])):
            predict_X[ind_x] = tuple(curr_x)
        sorted_i = np.argsort(dataset_X, axis=0, order = names)
        sorted_j = np.argsort(predict_X, axis=0, order = names)
        ind_i, ind_j = 0, 0
        count = 0
        while (ind_i != self.X_.shape[0] and
               ind_j != X.shape[0]):
            if np.equal(self.X_[sorted_i[ind_i]],
                        X[sorted_j[ind_j]]).all():
                count += 1
                y[sorted_j[ind_j]] = self.y_two_way_[sorted_i[ind_i]]
                ind_j += 1
            elif (self.X_[sorted_i[ind_i]] < X[sorted_j[ind_j]]).any():
                if not((X[sorted_j[ind_j]] < self.X_[sorted_i[ind_i]]).any()):
                    ind_i += 1
                elif (np.min(np.argmax(self.X_[sorted_i[ind_i]] < X[sorted_j[ind_j]])) <
                      np.min(np.argmax(X[sorted_j[ind_j]] < self.X_[sorted_i[ind_i]]))):
                    ind_i += 1
                else:
                    ind_j += 1
            else:
                ind_j += 1
        return y
           
    def __assign_three_way(self, X, fitting=False):
        """ Implementation of the three-way assignment of X to the
            available clusters.

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

        # setting hard clustering reference results
        # y will contain the result of hard clustering algorithm on currently
        #   predicting dataset
        # y_train will contain the result of hard clustering algorithm
        #   on fitting dataset
        if fitting:
            y = self.__spectral(X)
            X_train, y_train = X, y
            self.y_two_way_ = y
        else:
            y = self.__assign_two_way(X)
            X_train, y_train = self.X_, self.y_two_way_
        # Step 2. for each object v, compute its q-nearest neighborhood
        q_neighbors = []
        # to use if self.q > (n_samples) (it shouldn't happen)
        curr_q = min(self.q, X_train.shape[0]-1)
        if curr_q == 0:
            if fitting:
                self.dist_i_ = np.zeros(shape=(self.n_clusters))
            return y
        # computing pairwise euclidean distance in order to get neighbors' list
        X = X.real
        X_train = X_train.real
        dist = cdist(X, X_train)
        neighbors_list = np.argpartition(dist, curr_q, axis = 1)[:,:curr_q+1]
        if fitting:
            # not taking the first neighbor of each object because it would
            # be the object itself
            q_neighbors = neighbors_list[:, 1:curr_q+1]
        else:
            for (curr_x, ind_x) in zip(X, range(X.shape[0])):
                if (curr_x == X_train).all(axis=1).any():
                    # not taking the first neighbor because it's X[ind_x]   
                    q_neighbors.append(neighbors_list[ind_x, 1:curr_q+1])
                else:
                    q_neighbors.append(neighbors_list[ind_x, :curr_q])
        # y_result will contain the three-way clustering results
        y_result = np.zeros(shape=(y.shape[0], y.shape[1], 2))
        q_neighbors = np.array(q_neighbors)        
        # Step 3. Given a cluster c_i, for each object v not in c_i, if
        # Neig_q(v) and c_i are not disjoint, then v is in Fr(c_i)
        q_neigh_list = np.copy(q_neighbors.flat)
        for ind in range(self.n_clusters):
            indexes = np.argwhere((y[:,ind] == 0.0) &
                                  (np.max(y_train[q_neigh_list,ind].reshape(X.shape[0],
                                                                            curr_q),
                                          axis=1) > 0.0))
            y_result[indexes,ind,1] = 1 
            
        # Step 4. Given a cluster c_i, for each object v in c_i, if Neig_q(v)
        # is not in c_i, then v is in Fr(c_i), otherwise...
        # computing the average distance d_j between v and its q-nearest
        # neighborhood and the mean d_i of all d_j for v in c_i
        dist_j = []
        dist_i = np.zeros(shape=(self.n_clusters))
        for (curr_y, ind_x) in zip(y, range(X.shape[0])):
            ind = np.argmax(curr_y)
            rep_x = np.tile(X[ind_x], (curr_q, 1))
            d_j = np.sum(np.linalg.norm(X_train[q_neighbors[ind_x]] - rep_x,
                                        axis = 1)) / curr_q
            dist_j.append(d_j)
            dist_i[ind] += d_j
        # computing the actual value of d_i (only in fitting)
        if fitting:
            dist_i /= np.sum(y, axis=0)
            self.dist_i_ = dist_i
        
        for ind in range(self.n_clusters):
            for (curr_y, ind_x) in zip(y, range(X.shape[0])):
                if curr_y[ind] > 0.0:
                    if np.min(y_train[q_neighbors[ind_x],ind]) < 1.0:
                        y_result[ind_x,ind,1] = 1.0
                    else:
                        # if d_j < rho * d_i, then v is in the c_i's core,
                        # otherwise in the frontier
                        if dist_j[ind_x]  < self.rho * self.dist_i_[ind]:
                            y_result[ind_x,ind,0] = 1.0
                        else:
                            y_result[ind_x,ind,1] = 1.0
        return y_result

        

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

        y = self.__assign_three_way(X, fitting=True)
        self.X_ = X
        self.y_ = y
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
        
        y_fit = self.__assign_three_way(X)
        
        return y_fit

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
