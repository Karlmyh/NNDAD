'''
Different NN Based Algorithms
'''
import numpy as np

def knn(X,tree,k,n,dim,vol_unitball):
    """Standard k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Sanjoy Dasgupta and Samory Kpotufe. Optimal rates for k-nn density and mode 
    estimation. Advances in Neural Information Processing Systems, 27, 2014.
    """
    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()
        
    distance_matrix,_=tree.query(X,k+1)
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density= np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim)) 
    else:
        log_density= np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim))

    return log_density
    
    
def wknn(X,tree,k,n,dim,vol_unitball):
    """Weighted k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Gérard Biau, Frédéric Chazal, David Cohen-Steiner, Luc Devroye, and Carlos 
    Rodríguez. A weighted k-nearest neighbor density estimate for geometric 
    inference. Electronic Journal of Statistics, 5(none):204 – 237, 2011. 
    doi: 10.1214/11-EJS606. URL https://doi.org/ 10.1214/11-EJS606.
    """
    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()

    distance_matrix,_=tree.query(X,k+1)
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density= np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,1:]**dim).sum(axis=1))
    else:
        log_density= np.log((k+1)*k/2/n/vol_unitball/(distance_matrix[:,:-1]**dim).sum(axis=1))
    
    return log_density
    
    
def aknn(X,tree,k,n,dim,vol_unitball,**kwargs):
    """Adaptive k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    k : int
        Number of neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    Args:
        **threshold_r : float
            Threshold paramerter in AKNN to identify tail instances. 
        **threshold_num : int 
            Threshold paramerter in AKNN to identify tail instances. 
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Puning Zhao and Lifeng Lai. Analysis of knn density estimation, 2020.
    """

    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()

    distance_matrix,_=tree.query(X,k+1)
    
    # identify tail instances
    mask=tree.query_radius(X, r=kwargs["threshold_r"], 
                           count_only=True)<kwargs["threshold_num"]
    
    # rule out self testing
    if (distance_matrix[:,0]==0).all():
        log_density=np.log(k/n/vol_unitball/(distance_matrix[:,k]**dim)*mask+1e-30)
    else:
        log_density=np.log(k/n/vol_unitball/(distance_matrix[:,k-1]**dim)*mask+1e-30)
        
    return log_density
    
def bknn(X,tree,n,dim,vol_unitball,**kwargs):
    """Balanced k-NN density estimation. 

    Parameters
    ----------
    X : array-like of shape (n_test, dim_)
        List of n_test-dimensional data points.  Each row
        corresponds to a single data point.
        
    tree_ : "KDTree" instance
        The tree algorithm for fast generalized N-point problems.
        
    kmax : int
        Number of maximum neighbors to consider in estimaton. 
        
    n : int
        Number of traning samples.
        
    dim : int
        Number of features.
        
    vol_unitball : float
        Volume of dim_ dimensional unit ball.
        
    Args:
        **kmax : int
            Number of maximum neighbors to consider in estimaton. 
        **C : float 
            Scaling paramerter in BKNN.
        **C2 : float 
            Threshold paramerter in BKNN.
     

    Returns
    -------
    log_density: array-like of shape (n_test, ).
        Estimated log-density of test samples.
        
    Reference
    ---------
    Julio A Kovacs, Cailee Helmick, and Willy Wriggers. A balanced approach 
    to adaptive probability density estimation. Frontiers in molecular 
    biosciences, 4:25, 2017.
    """

    if len(X.shape)==1:
        X=X.reshape(1,-1).copy()
    
    C2=kwargs["C2"]
    
    log_density=[]
    
    
    for x in X:
        
        distance_vec,_=tree.query(x.reshape(1,-1),kwargs["kmax"])
        distance_vec=distance_vec[0]
        if distance_vec[0]==0:
            distance_vec=distance_vec[1:]
        k_temp=1
       
        while distance_vec[k_temp-1]**dim*k_temp<C2 and k_temp<kwargs["kmax"]-1:
            k_temp+=1
            
  
   
        log_density.append(np.log(k_temp*kwargs["C"]/n/vol_unitball/(distance_vec[k_temp-1]**dim)+1e-30))
   
        
    return np.array(log_density)
    
    