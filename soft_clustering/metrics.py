import numpy as np
from heapdict import heapdict
from scipy.spatial.distance import cdist

def __meet(y1, y2):
    """ Implementation of meet operation between two orthopartitions proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation"

    Parameters
    ----------
    y1 : array-like with shape (n_samples, n_cluster)
        First operand (orthopartition) of the meet function.
    y2 : array-like with shape (n_samples, n_cluster)
        Second operand (orthopartition) of the meet function.

    Returns
    -------
    y_meet : array-like with shape (n_samples, n_cluster)
        The orthopartition obtained as result of (y1 meet y2)
    
    """
    y_meet = []
    for ind_i in range(y1.shape[1]):
        for ind_j in range(y2.shape[1]):
            y_meet.append(y1[:, ind_i] * y2[:, ind_j])
    y_meet = np.array(y_meet).T 
    return y_meet

def __calc_entropy(y):
    """ Implementation of computation of logical entropy of a partition

    Parameters
    ----------
    y : array-like with shape(n_samples, n_cluster)
        Input partition, which is requested to compute its logical entropy.

    Returns
    -------
    entropy: float
        The entropy value of the partition (y)

    """
    # computing the size of the universe, hence the number of samples
    u_size = y.shape[0]
    # computing the size of dit(y)
    dit_size = 0
    for block in range(y.shape[1]):
        block_size = np.sum(y[:,block])
        dit_size += (u_size - block_size) * block_size
    entropy = dit_size / (u_size ** 2)
    return entropy

def __up_entropy(y):
    """ Implementation of Polynomial Upper Logical Entropy (P-ULE) proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation".

        Actually this implementation has an average time of
        O(max(|U| * n, |U| * log(n))) that we can reduce to O(|U| * n)
        but the worst case would be O(max(|U|^2 * n, |U| * log(n))), since we use n
        hash tables with size in the order of O(|U|), we can reduce to
        O(|U|^2 * n).

    Parameters
    ----------
    y : array-like with shape (n_samples, n_cluster)
        Input orthopartition, which is requested to compute
        its upper logical entropy.

    Returns
    -------
    up_entropy: float
        Upper Logical Entropy of orthopartition y.
    
    """
    priority_queue = heapdict()
    y = np.copy(y)
    # bnd_all[x] = 1 if x-th object is in boundary regions
    bnd_all = np.where(np.sum(y, axis=1) > 1.0, 1, 0)
    bnd_size = np.sum(bnd_all)
    # bnd[:,i] contains the i-th cluster's boundary
    bnd = y * bnd_all.reshape(y.shape[0], 1)
    # creating bnd sets where
    # bnd_set[k][x] is in the x-th set if x-th object is in k-th cluster's
    # boundary
    bnd_set = [set() for x in range(y.shape[1])]
    for curr_sample in range(y.shape[0]):
        for (sample_value, ortho_i) in zip(bnd[curr_sample], range(y.shape[1])):
            if sample_value > 0:
                bnd_set[ortho_i].add(curr_sample)
    # since heapdict use a min heap as priority-queue, we can
    # use directly |P_i| as parameter to sort
    # computing P_i for each orthopair and its size
    for ortho_i in range(y.shape[1]):
        priority_queue[ortho_i] = np.sum(y[:, ortho_i] * (1 - bnd_all))
    while bnd_size > 0:
        ortho_i, size = priority_queue.popitem()
        # considering only bnd objects
        obj_removed = 0
        if len(bnd_set[ortho_i]) > 0:
            obj_removed = list(bnd_set[ortho_i])[0] 
        # if there's no element in Bnd_i (obj_removed actually had a value of
        # y[obj_removed,ortho_i] * bnd[obj_removed] == 0, then we can just
        # ignore this iteration and don't insert again ortho_i
        # in priority_queue
        if (y[obj_removed,ortho_i] * bnd_all[obj_removed] > 0):
            for ortho_j in range(y.shape[1]): 
                if (ortho_j != ortho_i and
                    y[obj_removed][ortho_j] > 0):
                    y[obj_removed][ortho_j] = 0
                    bnd_set[ortho_j].remove(obj_removed)
            # removing from bnd the boundary object that has been
            # located in P_i
            bnd_set[ortho_i].remove(obj_removed)
            bnd_all[obj_removed] = 0
            bnd_size -= 1
            priority_queue[ortho_i] = size + 1
    up_entropy = __calc_entropy(y)
    return up_entropy


def __low_entropy(y):
    """ Implementation of Polynomial Lower Logical Entropy (P-LLE) proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation".

        Actually this implementation is O((|U| + n) * n * log(n)), but since
        n <= |U|, we can use O(|U| * n * log(n)).

    Parameters
    ----------
    y : array-like with shape (n_samples, n_cluster)
        Input orthopartition, which is requested to compute
        its lower logical entropy.

    Returns
    -------
    low_entropy: float
        Lower Logical Entropy of orthopartition y.

    """
    priority_queue = heapdict()
    y = np.copy(y)
    # since heapdict use a min heap as priority-queue, we use
    # |N_i| as parameter to sort instead of |P_i U Bnd_i|
    removed = {}
    
    for ortho_i in range(y.shape[1]):
        priority_queue[ortho_i] = y.shape[0] - np.sum(y[:, ortho_i])
        removed[ortho_i] = False

    # bnd_all[x] = 1 if x-th object is in boundary regions
    bnd_all = np.where(np.sum(y, axis=1) > 1.0, 1, 0)
    bnd_size = np.sum(bnd_all)
    # bnd[:,i] contains the i-th cluster's boundary
    bnd = y * bnd_all.reshape(y.shape[0], 1)
    while bnd_size > 0: 
        ortho_i, size = priority_queue.popitem()
        
        removed[ortho_i] = True
        obj_removed = []
        for obj in range(bnd.shape[0]):
            if bnd[obj, ortho_i] > 0.0:
                obj_removed.append(obj)
                # removing from bnd_all the boundary objects that are in ortho_i
                bnd_all[obj] = 0
                bnd_size -= 1
        
        for ortho_j in range(y.shape[1]):
            # not considering the orthopairs already popped
            # from the priority_queue
            if not(removed[ortho_j]):
                # computes the two operations on Bnd_i and N_i simultaneously
                count_bnd = 0
                for obj in obj_removed:
                    # if obj is in ortho_j's boundary, it gets removed from it
                    if y[obj, ortho_j] > 0:
                        y[obj, ortho_j] = 0
                        count_bnd += 1
                priority_queue[ortho_j] += count_bnd
    low_entropy = __calc_entropy(y)
    return low_entropy

def __conv_to_ortho(y):
    """ Implementation of conversion function to orthopartition
        proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation"


    Parameters
    ----------
    y : array-like with possible shapes {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The input result to convert into orthopartition.

    Returns
    -------
    y_ortho : array-like with shape (n_samples, n_cluster)
        The orthopartition that represents the input result.

    """
    
    y_ortho = np.copy(y)
    if len(y_ortho.shape) == 1:
        # if y is a Hard clustering result
        n_samples = y_ortho.shape[0]
        n_clusters = (np.max(y_ortho) + 1).astype('int32')
        y_res = np.zeros(shape=(n_samples, n_clusters))
        for (curr_y, curr_res) in zip(y_ortho, y_res):
            curr_res[int(curr_y)] = 1
        y_ortho = y_res
    elif len(y_ortho.shape) == 3:
        # if y is a Three-Way Clustering result: add a new cluster
        # with empty lower bound and upper bound containing all the
        # boundary objects that are mapped into only one fringe region
        # making the Three-way Clustering result into a orthopartition
        to_add = y_ortho[:,:,1]
        new_cluster = np.zeros(shape=(y_ortho.shape[0], 1, 2))
        new_cluster[:,0,1] = np.where(np.sum(to_add, axis=1) == 1, 1, 0)
        # if there are no boundary objects mapped into only one fringe region
        if np.sum(new_cluster) > 0.0:
            y_ortho = np.append(y_ortho, new_cluster, axis=1)
        y_ortho = np.sum(y_ortho, axis=2)
        
    return y_ortho

def n_sami_index(y, y_pred):
    """ Implementation of the evaluation function of N-SAMI Index
        proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation"

    Parameters
    ----------
    y : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The actual labels of the samples.
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.
        
    Returns
    -------
    n_sami : the Normalized Soft Average Mutual Information (N-SAMI) index value.
    
    """
    y_pred = __conv_to_ortho(y_pred)
    y = __conv_to_ortho(y)

    # computing mean entropy of rough clustering result
    h_low_pred = __low_entropy(y_pred)
    h_up_pred = __up_entropy(y_pred)
    h_mean_pred = (h_low_pred + h_up_pred) / 2

    # computing mean entropy of gold standard
    h_low_actual = __low_entropy(y)
    h_up_actual = __up_entropy(y)
    h_mean_actual = (h_low_actual + h_up_actual) / 2

    # computing mean entropy of meet between clustering result
    # and gold standard
    y_meet = __meet(y, y_pred)
    h_low_meet = __low_entropy(y_meet)
    h_up_meet = __up_entropy(y_meet)
    h_mean_meet = (h_low_meet + h_up_meet) / 2
    
    # using entropies in order to compute N-SAMI Index
    n_sami = ((h_mean_actual + h_mean_pred - h_mean_meet) /
              min(h_mean_actual, h_mean_pred))
    return n_sami

def __general_entropy(y):
    """ Implementation of Fast General Entropy (FGE) proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation".

        This implementation is theta(n * |U| ^ 2)


    Parameters
    ----------
    y : array-like with shape (n_samples, n_cluster)
        Input orthopartition, which is requested to compute
        its general entropy.

    Returns
    -------
    gen_entropy: float
        General Entropy of orthopartition y.

    """
    
    res = 0
    # in order to compute value of Res, we need to consider
    # each pair of samples <s_i, s_j> such that i < j
    for sample_i in np.arange(y.shape[0]):
        size_j = y.shape[0] - sample_i - 1
        rep_i = np.tile(y[sample_i], (size_j, 1))
        y_j = y[sample_i+1:]
        # computing value of N for all the pairs <s_i, s_j> such that i < j
        n = (np.sum(rep_i * y_j, axis=1) /
             (np.sum(rep_i, axis=1) * np.sum(y_j, axis=1)))
        res += np.sum(1 - n)
    gen_entropy = res / (y.shape[0] ** 2)
    
    return gen_entropy


def n_slmi_index(y, y_pred):
    """ Implementation of the evaluation function of N-SLMI Index
        proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation"

    Parameters
    ----------
    y : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The actual labels of the samples.
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.
        
    Returns
    -------
    n_slmi : Normalized Soft Logical Mutual Information (N-SLMI) index value.
            
    """
    y_pred = __conv_to_ortho(y_pred)
    y = __conv_to_ortho(y)

    # computing general entropy of rough clustering result
    h_pred = __general_entropy(y_pred)

    # computing general entropy of gold standard
    h_actual = __general_entropy(y)
    
    # computing general entropy of meet between clustering result
    # and gold standard
    y_meet = __meet(y, y_pred)
    
    h_meet = __general_entropy(y_meet)
    
    # using entropies in order to compute N-SLMI Index
    n_slmi = ((h_actual + h_pred - h_meet) /
              min(h_actual, h_pred))
    return n_slmi

def soft_purity_index(y, y_pred):
    """ Implementation of the evaluation function of Soft Purity (SP)
        proposed in:
        "Andrea Campagner, Davide Ciucci, Orthopartitions and soft
        clustering: Soft mutual information measures for clustering validation"

    Parameters
    ----------
    y : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The actual labels of the samples.
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.
        
    Returns
    -------
    soft_pur : The Soft Purity (SP) index value.

    """
    y_pred = __conv_to_ortho(y_pred)
    y = __conv_to_ortho(y)
    
    n_samples = y.shape[0]

    # y_pred_count[x] counts how many times y_pred[x,:] = 1
    y_pred_count = np.sum(y_pred, axis=1)
    # y_count[x] counts how many times y[x,:] = 1
    y_count = np.sum(y, axis=1)

    # bnd_pred_all[x] = 1 if x-th object is in clusters' boundary regions
    bnd_pred_all = np.where(y_pred_count > 1.0, 1, 0)
    # bnd_pred[:, i] contains the i-th cluster's boundary
    # pos_pred[:, i] contains the i-th cluster's positive region
    bnd_pred = y_pred * bnd_pred_all.reshape(y_pred.shape[0], 1)
    pos_pred = y_pred * (1 - bnd_pred_all.reshape(y_pred.shape[0], 1))

    # bnd[x] = 1 if x-th object is in class' boundary regions
    bnd = np.where(y_count > 1.0, 1, 0)

    soft_pur = 0.0
    for ortho_i in range(y.shape[1]):
        
        bnd_i = y[:,ortho_i] * bnd
        pos_i = y[:,ortho_i] * (1 - bnd)
        rep_pos_i = np.tile(pos_i, (y_pred.shape[1], 1)).T
        rep_bnd_i = np.tile(bnd_i, (y_pred.shape[1], 1)).T
        # considers overlap between P_i and P_j for each cluster C_j
        prob = np.sum(rep_pos_i * pos_pred, axis=0)
        # considers overlap between Bnd_i and P_j for each cluster C_j
        prob += np.sum(((rep_bnd_i * pos_pred) /
                        y_count.reshape(y.shape[0], 1)),
                       axis=0)
        # considers overlap between P_i and Bnd_j for each cluster C_j
        prob += np.sum(((rep_pos_i * bnd_pred) /
                        y_pred_count.reshape(y.shape[0], 1)),
                       axis=0)
        # considers overlap between Bnd_i and Bnd_j for each cluster C_j
        prob += np.sum(((rep_bnd_i * bnd_pred) /
                        (y_pred_count * y_count).reshape(y.shape[0], 1)),
                       axis=0)
        # brings the maximum value obtained with the i-th orthopartition
        # and a cluster C_j
        soft_pur += np.max(prob)
        
    soft_pur /= n_samples
    return soft_pur

def pi_r_dbi(X, centroids, y_pred):
    """ Implementation of the Pi-rough Davies-Bouldin Index proposed in:
        "Peters, G.: Rough clustering utilizing the principle of indifference."

    Parameters
    ----------
    X : array-like with shape(n_samples, n_features)
        The input samples that have been clustered.
    centroids : array-like with shape (n_cluster, n_features)
        The result centroids obtained by clustering X:
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.
        
    Returns
    -------
    r_dbi : the Pi-Rough Davies-Bouldin Index value.
            
    """
    # translate 2-Way Clustering result into an orthopartition
    if len(y_pred.shape) == 1:
        y_pred = __conv_to_ortho(y_pred)
    elif len(y_pred.shape) == 3:
        # since we need the centroids of each cluster, we can't translate
        # 3-Way Clustering result to orthopartition using __conv_to_ortho
        # algorithm because we would have another cluster without a centroid
        y_pred = np.sum(y_pred, axis=2)
    r_dbi = 0.0
    n_cluster = centroids.shape[0]

    # computing intra-cluster distance for each cluster
    intra_num = np.zeros(shape=(n_cluster))
    intra_den = np.zeros(shape=(n_cluster))
    all_dist = cdist(X, centroids)
    for k in range(n_cluster):
        intra_num[k] = np.sum(np.where(y_pred[:, k] > 0.0,
                                       all_dist[:, k] / np.sum(y_pred, axis=1),
                                       0.0))
        intra_den[k] = np.sum(np.where(y_pred[:, k] > 0.0,
                                       1 / np.sum(y_pred, axis=1),
                                       0.0))
    intra_dist = intra_num / intra_den
    
    # Computing Davies Bouldin Index
    for (ind_k, cen_k) in zip(range(n_cluster), centroids):
        db_values = [(intra_dist[ind_i] + intra_dist[ind_k]) / np.linalg.norm(cen_k - cen_i)
                     for (ind_i, cen_i) in zip(range(n_cluster), centroids)
                     if ind_i != ind_k and not np.equal(cen_k, cen_i).all()]
        if len(db_values) > 0:
            r_dbi += np.max(np.array(db_values))
        
    r_dbi /= n_cluster
    return r_dbi


def r_rand_index(y, y_pred):
    """ Implementation of the evaluation function of R-Rand Index
        proposed in:
        "Depaolini M.R., Ciucci D., Calegari S., Dominoni M. (2018)
        External Indices for Rough Clustering."

    Parameters
    ----------
    y : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The actual labels of the samples.
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.

    Returns
    -------
    r_rand : The R-rand index value.

    """
    if len(y_pred.shape) < 3:
        # converting not three-way results into orthopartition
        y_pred = __conv_to_ortho(y_pred)
    else:
        # converting three-way clustering results into rough results using
        # Principle of Indifference
        y_pred = np.sum(y_pred, axis=2)
    y_actual = __conv_to_ortho(y)
    
    # computing the membership degree of input samples
    # since the prediction is represented as an orthopartition
    y_pred_prob = y_pred / np.sum(y_pred, axis=1).reshape(y_pred.shape[0],1) 

    # constructing prediction matrices B_i as D_i * D_i(T)
    b_pred = np.matmul(y_pred_prob, y_pred_prob.T)
    b_actual = np.matmul(y_actual, y_actual.T)
    wf_a, wf_b, wf_c, wf_d = 0.0, 0.0, 0.0, 0.0
    for j in np.arange(1, y.shape[0]):
        b_p = b_pred[j]
        b_a = b_actual[j]
        b_p_n = 1 - b_p
        b_a_n = 1 - b_a
        wf_a += np.inner(b_p, b_a)
        wf_b += np.inner(b_p, b_a_n)
        wf_c += np.inner(b_p_n, b_a)
        wf_d += np.inner(b_p_n, b_a_n)

    if (np.isclose(wf_a, wf_b) and
        np.isclose(wf_b, wf_c) and
        np.isclose(wf_c, wf_d) and
        np.isclose(wf_d, 0.0)):
        return 0
    r_rand = (wf_a + wf_d) / (wf_a + wf_b + wf_c + wf_d)
    return r_rand


def rand_index(y, y_pred):
    """ Implementation of the evaluation function of Rand Index
        on lower boundary's objects

    Parameters
    ----------
    y : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The actual labels of the samples.
    y_pred : array-like with possible shapes: {(n_samples), (n_samples, n_cluster),
        (n_samples, n_cluster, 2)}
        The predicted values.

    Returns
    -------
    rand : The R-rand index value.
    coverage : Value in range [0; 1] that represents the data coverage
            of the index (expressed as the lower bound ratio of the
            predicted clustering).

    """
    if len(y_pred.shape) < 3:
        # converting not three-way results into orthopartition
        y_pred = __conv_to_ortho(y_pred)
    else:
        # converting three-way clustering results into rough results using
        # Principle of Indifference
        y_pred = np.sum(y_pred, axis=2)
    y_actual = __conv_to_ortho(y)
    
    # computing the membership degree of input samples
    # since the prediction is represented as an orthopartition
    y_pred_prob = y_pred / np.sum(y_pred, axis=1).reshape(y_pred.shape[0],1) 

    # keeping only the samples that are assigned to a lower bound
    y_pred_lower = np.array([sample for sample in y_pred if np.sum(sample) <= 1])
    y_lower = np.array([sample for (sample, sample_pred) in zip(y_actual, y_pred)
               if np.sum(sample_pred) <= 1])
    
    # R-Rand index of two-way clustering has the same result of a
    # Rand index considering the same clustering.
    if len(y_lower) == 0:
        return (0, 0)
    return (r_rand_index(y_lower, y_pred_lower), len(y_pred_lower) / len(y_pred))

