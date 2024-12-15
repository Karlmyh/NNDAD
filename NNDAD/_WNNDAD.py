"""
Nearest Neighbor Distance Anomaly Detection
-------------------------------------------
"""

import numpy as np
import math
from sklearn.neighbors import KDTree
from time import time
from numba import njit







class WNNDAD(object):
    """Regularized Nearest Neighbor Density for Anomaly Detection.

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
        
    max_samples_ratio : float in [0,1], default = 1.
        Portion of samples to consider when solving the optimization problem. 
        
    bagging_round : int, default = 1.
        Potential bagging round, influence the the optimization object.
        
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
        metric = "euclidean",
        leaf_size = 40,
        alpha = 1,
        k = 2,
        use_knn = False
    ):
        self.metric = metric
        self.leaf_size = leaf_size
        self.alpha = alpha
        self.k = k
        self.use_knn = use_knn

    def fit(self, X, y = None):
        

        self.tree_ = KDTree(
            X,
            metric = self.metric,
            leaf_size = self.leaf_size,
        )
        
        self.n_train_, self.dim_ = X.shape
        self.vol_unitball_ = math.pi**(self.dim_/2)/math.gamma(self.dim_/2+1)

        if not self.use_knn:
            self.weights = np.array([ (i + 1)**(1 / self.alpha - 1) for i in range(self.k)])
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = np.zeros(self.k)
            self.weights[-1] = 1
        
        return self

    def predict(self, X, y = None):
        """Compute the weighted k nearest neighbor distance. 
        
        Parameters
        ----------
        X : array-like of shape (n_test, dim_)
            An array of points to query.  Last dimension should match dimension
            of training data (dim_).

        Returns
        -------
        distance : float
        
        """
        distance = self.tree_.query(X, self.k)[0]
        return (distance @ self.weights).ravel()
    
    def density(self, X, y = None):
        """Compute the density estimation corresponding to the optimized empirical risk 
        upper bound minimized weights.
        
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
        numerator = np.array([ ((i + 1)/ self.n_train_ )**(1 / self.dim_) * self.weights[i]
                  for i in range(self.k)])
        numerator = numerator.sum()**self.dim_

        return numerator / self.vol_unitball_ / self.predict(X)**self.dim_
    
   


        
    
    

    

