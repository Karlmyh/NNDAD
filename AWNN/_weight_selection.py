'''
Weight Selection
----------------
'''

import numpy as np
import math
from numba import njit

@njit
def weight_selection(beta,cut_off):
    """Find the optimization solution of optimal weights. 

    Parameters
    ----------
    Beta : array-like of shape (potentialNeighbors, )
        Array of rescaled distance vector. Suppose to be increasing.
        
    cut_off : int
        Number of neighbors for cutting AWNN to KNN. 

    Returns
    -------
    estAlpha: array-like of shape (potentialNeighbors, )
        Solved weights. 
        
    alphaIndexMax: int
        Solved number of neighbors.
        
    Reference
    ---------
    Oren Anava and Kfir Levy. k*-nearest neighbors: From global to local. 
    Advances in neural information processing systems, 29, 2016.
    """
    
    potentialNeighbors=len(beta)
    alphaIndexMax=0
    lamda = beta[0]+1 
    Sum_beta = 0
    Sum_beta_square = 0

    # iterates for k
    
    while ( lamda>beta[alphaIndexMax] ) and (alphaIndexMax<potentialNeighbors):
        # update max index
        alphaIndexMax +=1
        # updata sum beta and sum beta square
        Sum_beta += beta[alphaIndexMax-1]
        Sum_beta_square += (beta[alphaIndexMax-1])**2
        
        # calculate lambda
        
        if  alphaIndexMax  + (Sum_beta**2 - alphaIndexMax * Sum_beta_square)>=0:
                
            
            lamda = (1/alphaIndexMax) * ( Sum_beta + math.sqrt( alphaIndexMax  + (Sum_beta**2 - alphaIndexMax * Sum_beta_square) ) )
            
        else:
            alphaIndexMax-=1
            break
            
    if lamda>100000000:
        return np.zeros(potentialNeighbors),0
    
    # estimation
    estAlpha=np.zeros(potentialNeighbors)

    
    if alphaIndexMax<cut_off:
        estAlpha[cut_off-1]=1
        return estAlpha,cut_off
    
    
    for j in range(alphaIndexMax):
        estAlpha[j]=lamda-beta[j]
    
    
    estAlpha=estAlpha/np.linalg.norm(estAlpha,ord=1)
    
    return estAlpha,alphaIndexMax




        
    