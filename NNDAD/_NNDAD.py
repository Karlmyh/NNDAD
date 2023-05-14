"""
Nearest Neighbor Distance Anomaly Detection
-------------------------------------------
"""

import numpy as np
import math
from sklearn.neighbors import KDTree
from time import time
from numba import njit

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


class NNDAD(object):
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
        max_samples_ratio = 1.,
        
    ):
        self.lamda_list = lamda_list
        self.metric = metric
        self.leaf_size = leaf_size
        self.max_samples_ratio = max_samples_ratio
        


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
        self.tree_ = KDTree(
            X,
            metric = self.metric,
            leaf_size = self.leaf_size,
        )
        
        time_e = time()
        print("kd-tree time {}s".format(time_e - time_s))
        time_s = time()
        
        self.dim_ = X.shape[1]
        self.n_train_ = X.shape[0]
        self.vol_unitball_ = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)

        
        self.mean_k_distance_train = self.tree_.query(X, int(self.max_samples_ratio * self.n_train_))[0].mean(axis = 0)
        
        
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
        for key in ['lamda',"max_samples_ratio"]:
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

        weighted_distance = (self.tree_.query(X, self.n_train_)[0] @ self.weights).ravel()
        return weighted_distance
    
    
    def density(self,X, y = None):
        
        vol_ball = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)
        numerator = np.sum(np.array([ ((i + 1)/ self.n_train_ )**self.dim_ for i in range(self.n_train_)]) * self.weights )**self.dim_
        return numerator / vol_ball / (self.tree_.query(X, self.n_train_)[0] @ self.weights).ravel()
    
   
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
        
        
        
                
                
            
                
        
        
        
    
    

    

