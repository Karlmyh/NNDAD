import numbers
import math
import numpy as np
from time import time
from sklearn.neighbors import KDTree
from sklearn.utils import resample
from joblib import Parallel, delayed

from ._NNDAD import NNDAD



def base_fit(X, max_samples, random_seed, lamda_list, bagging_round):
    resample_X = resample(X,
                          replace=False,
                          n_samples=max_samples,
                          random_state=random_seed)
    estimator = NNDAD(lamda_list=lamda_list,
                      bagging_round = bagging_round,
                     )
    estimator.fit(resample_X)
    return estimator


def base_predict(X, model):
    return model.predict(X).reshape(-1, 1)


class BNNDAD(object):
    def __init__(self,
                 n_estimators,
                 max_samples,
                 lamda_list,
                 random_state=None,
                 ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.lamda_list = lamda_list
        self.random_state = random_state

    def fit(self, X):
        # Validate max_samples
        if not isinstance(self.max_samples, numbers.Integral):
            self.max_samples = int(self.max_samples * X.shape[0])
        if not (0 < self.max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")
        # about random seed
        random_state = np.random.RandomState(self.random_state)
        random_seed_seq = []
        UINT_MAX = np.iinfo(np.uint32).max
        for _ in range(self.n_estimators):
            random_seed_seq.append(random_state.randint(0, UINT_MAX))
        self.estimators = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(base_fit)(X, self.max_samples, random_seed_seq[idx],
                              self.lamda_list, self.n_estimatorsn_jobs)
            for idx in range(self.n_estimators))

    def predict(self, X):
        score = np.zeros(shape=(X.shape[0], 1), dtype=np.float64)
        for idx in range(self.n_estimators):
            score += base_predict(X, self.estimators[idx])
        score /= self.n_estimators
        return score
    
    
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
        for key in ['lamda_list',"metric", "leaf_size",
                   'max_samples_ratio', 'bagging_round']:
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

