import numpy as np
from distributions import TDistribution,MultivariateNormalDistribution,MixedDistribution


__all__ = ['mc_sampling']

def mc_sampling(X,nsample=1,**kwargs):
    
    dim=X.shape[1]
    
    if kwargs["method"]=="bounded":
        lower=np.array([np.quantile(X[:,i],kwargs["ruleout"]) for i in range(dim)])
        upper=np.array([np.quantile(X[:,i],1-kwargs["ruleout"]) for i in range(dim)])
            
        np.random.seed(kwargs["seed"])
        return np.random.rand(int(nsample),dim)*(upper-lower)+lower,np.ones(int(nsample))/np.prod(upper-lower)
    if kwargs["method"]=="heavy_tail":
        density=TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    if kwargs["method"]=="normal":
        density=MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T))))
        np.random.seed(kwargs["seed"])
        return density.generate(int(nsample))
    
    if kwargs["method"]=="mixed":
        density1 = MultivariateNormalDistribution(mean=X.mean(axis=0),cov=np.diag(np.diag(np.cov(X.T)))) 
        density2 = TDistribution(loc=np.zeros(dim),scale=np.ones(dim),df=2/3)
        
    
        density_seq = [density1, density2]
        prob_seq = [0.7,0.3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix.generate(int(nsample))