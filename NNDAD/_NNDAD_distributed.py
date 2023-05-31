"""
Nearest Neighbor Distance Anomaly Detection
-------------------------------------------
"""

import numpy as np
import math
from sklearn.neighbors import KDTree
from time import time
from numba import njit
from sklearn.model_selection import KFold
from multiprocessing import Pool

def single_parallel(input_tuple):
    kdtree, X, query_num = input_tuple
    dist_vec = kdtree.query(X, query_num)[0].mean(axis = 0)
    return dist_vec

def single_parallel_matrix(input_tuple):
    kdtree, X, query_num, weights = input_tuple
    dist_matrix = kdtree.query(X, query_num)[0]
    dist_vec = np.array([(dist_vector * weights).sum() for dist_vector in dist_matrix])
    return dist_vec

@njit
def compute_weights(beta):
    potential_neighbors = len(beta)
    current_index = 0
    multiplier = beta[0]+1 
    sum_beta = 0
    sum_beta_square = 0

    while ( current_index < potential_neighbors - 1 ) and ( multiplier > beta[current_index] ):
        current_index +=1

        sum_beta += beta[current_index - 1]
        sum_beta_square +=  beta[current_index - 1]**2

        if  current_index  + (sum_beta**2 - current_index * sum_beta_square) >= 0:
            multiplier =  ( sum_beta + math.sqrt( current_index  + (sum_beta**2 - current_index * sum_beta_square) ) ) 
            multiplier /= current_index
        else:
            current_index -= 1
            break

    estimated_weights = np.zeros(potential_neighbors)

    for j in range(current_index):
        estimated_weights[j] = multiplier - beta[j]


    estimated_weights = estimated_weights / np.linalg.norm(estimated_weights, ord = 1)
    
    return estimated_weights


class NNDADDIST(object):
    """NNDAD

    Read more in Nearest Neighbor Distance Anomaly Detection

    Parameters
    ----------
    lamda_list : list, default=[1.0].
        The potential tuning parameter in Empirical Risk Upper Bound Minimization 
        which controls the optimized weights. 

    metric : str, default='euclidean'.
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        'sklearn.KDTree' for a description of available algorithms. 
        Default is 'euclidean'.
        
    leaf_size : int, default = 40.
        KDTree parameters. 
        
    max_samples_ratio : float in [0,1], default = 1
        Portion of distances to consider. 
        
    Attributes
    ----------
    n_train_ : int
        Number of training instances.

    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.

    dim_ : int
        Number of features.

    vol_unitball_ : float
        Volume of dim_ dimensional unit ball.

    weights : array-like of shape (n_train_, ).
        Estimated weights of nearest_distances.

    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.

    
    """

    def __init__(
        self,
        *,
        lamda_list = [1.0],
        metric = "euclidean",
        leaf_size = 40,
        distributed_fold = 2,
        thred_num = 5
    ):
        self.lamda_list = lamda_list
        self.metric = metric
        self.leaf_size = leaf_size
        self.distributed_fold = distributed_fold
        self.thred_num = thred_num

    def fit(self, X, y = None):
        """Fit the NNDAD on the data.

        Parameters
        ----------
        X : array-like of shape (n_train_, dim_)
            Array of dim_-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
            
     
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        time_s = time()
        # drop redundant points
        X_divide = X[:int((X.shape[0] // self.distributed_fold ) * self.distributed_fold),:]
        # divide the samples
        kfolder = KFold(n_splits = self.distributed_fold)
        X_divide_list = [X_divide[test_index] for i, (_, test_index) in enumerate(kfolder.split(X_divide))]
        self.X_divide_list = X_divide_list
        
        
        self.tree_list = []
        for X_divide_single in X_divide_list:
            tree_ = KDTree(
            X_divide_single,
            metric = self.metric,
            leaf_size = self.leaf_size,
        )
            self.tree_list.append(tree_)
            
  
        
        time_e = time()
        print("kd-tree time {}s".format(time_e - time_s))
        time_s = time()
        
        self.dim_ = X.shape[1]
        self.n_train_ = X.shape[0]
        self.vol_unitball_ = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        

        self.mean_k_distance_train = np.zeros( self.n_train_ // self.distributed_fold)
        X_list = [(self.tree_list[i], X_divide_list[i], self.n_train_ // self.distributed_fold) for i in range(self.distributed_fold)]
        with Pool(self.thred_num) as p:
            dist_list = p.map(single_parallel, X_list)
                
        for dist_vec in dist_list:
            self.mean_k_distance_train += dist_vec
        self.mean_k_distance_train /= self.n_train_
        
        time_e = time()
        print("query time {}s".format(time_e - time_s))
        time_s = time()
        
        
        
        self.score_vec = []
        min_score = 1e10
        for lamda in self.lamda_list:
            weights =  self.compute_weights( self.mean_k_distance_train / lamda )
            score = (weights * self.mean_k_distance_train).sum() + np.sqrt(np.log(self.n_train_)) * np.linalg.norm(weights)
            if score < min_score:
                self.best_score = score
                self.lamda = lamda
                self.weights = weights
                min_score = score 
            self.score_vec.append(score)
        
        time_e = time()
        print("optimization of weights time {}s".format(time_e - time_s))
        time_s = time()
        
        return self
    
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['lamda',"max_samples_ratio", "distributed_fold"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    
    def compute_weights(self, beta):
        
        return compute_weights(beta)
        
    

    
    
    
    def predict(self, X, y = None):
        """Compute the weighted k nearest neighbor distance. 
        
        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).

        Returns
        -------
        distance 
        
        """
        
       
        self.train_sample_score = np.zeros(X.shape[0])
        kfolder = KFold(n_splits = self.distributed_fold)
        
        predict_list = [(self.tree_list[i], X, self.n_train_ // self.distributed_fold, self.weights) for i, (_, test_index) in enumerate(kfolder.split(X))]
        with Pool(self.thred_num) as p:
            dist_list = p.map(single_parallel_matrix, predict_list)
                
                
        for dist_vec in dist_list:
            self.train_sample_score +=  dist_vec 
        
       
        return self.train_sample_score
    
    

    
   
    def score(self, X, y=None):
        """Compute the ERUBM

        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            List of n_test-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        score : float
            
        """
        
        return self.best_score
    
    def ERUB(self, X):
        '''
        return the erub of X
        '''
        
        return self.predict(X) + np.linalg.norm(self.weights) * np.sqrt(np.log(self.weights.shape[0]))
        
        
        
#     def get_train_score(self):
#         return (self.mean_k_distance_train * self.weights).sum()
                
            
                
        
        
        
    
    

    

