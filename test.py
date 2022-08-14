from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import GridSearchCV

from AWNN import AWNN
from KNN import KNN 
from AKDE import AKDE
from sklearn.neighbors import KernelDensity

from synthetic_distributions import TestDistribution

# setting s
dim=2
n_train=100
n_test=100
distribution=2
np.random.seed(1)


# generate data
density=TestDistribution(distribution,dim).returnDistribution()
X_train, pdf_X_train = density.generate(n_train)
X_test, pdf_X_test = density.generate(n_test)


## produce estimation
# KNN
parameters={"C":[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]}

cv_model_KNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=10,cv=10)
_=cv_model_KNN.fit(X_train)
cv_model_KNN=cv_model_KNN.best_estimator_
    

#print(model_KNN.predict(X_test))
est_BKNN=np.exp(cv_model_KNN.predict(X_test).reshape(-1,n_test))

print(np.abs(est_BKNN-pdf_X_test).mean())

model_AWNN=AWNN(C=1).fit(X_train)
#print(model_KNN.predict(X_test))
est_AWNN=np.exp(model_AWNN.predict(X_test).reshape(-1,n_test))
print(np.abs(est_AWNN-pdf_X_test).mean())