import os
import time
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from NNDAD import BNNDAD
from distribution import TestDistribution

# repeat times
iter_times = 20
# dimensions
dim_vec = [1,2, 3,4, 5, 6, 7, 8, 9, 10]
# number of samples
n_vec = [1000, 5000]
# ratio of anomalies
ratio_vec = [0.1]
# potential bagging vec
n_estimators_vec = [1,5,10]
# process number
n_jobs = 20

log_file_dir = "./results/simulation/"


for iteration in range(iter_times):
    for dim, n, ratio in product(dim_vec, n_vec, ratio_vec):

        distribution = TestDistribution(index = 5, dim = dim).returnDistribution()
        X, density = distribution.generate(n)
        threshold = np.quantile(density, ratio)
        y = (density < threshold).astype(int)
              
        # bagged nearest neighbor distance best
      
        for lamda, n_estimators in product(lamda_seq, n_estimators_vec):
            time_start = time.time()
            model = BNNDAD(
                           n_estimators=n_estimators,
                           random_state=101,
                           lamda_list=[lamda],
                           n_jobs = n_jobs
                         )
            model.fit(X)
            time_end = time.time()
            trainingtime = time_end - time_start
            time_start = time.time()
            y_pred = model.predict(X).reshape(-1,1)
            scaler = MinMaxScaler()
            y_pred = scaler.fit_transform(y_pred)
            time_end = time.time()
            testtime = time_end - time_start
            mae = np.abs(density - model.density(X)).mean()
            roc_auc = roc_auc_score(y, y_pred)
            log_file_name = "NNDAD-B.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "NNDAD-B", iteration, dim, n, ratio, 
                    trainingtime, testtime, roc_auc, mae, n_estimators, lamda )
                f.writelines(logs)
                
        # bagged nearest neighbor distance self tuning
        for n_estimators in n_estimators_vec:
            time_start = time.time()
            model = BNNDAD(
                           n_estimators=n_estimators,
                           lamda_list=lamda_seq,
                           random_state=101,
                           n_jobs = n_jobs,
                             )
            model.fit(X)
            time_end = time.time()
            trainingtime = time_end - time_start
            time_start = time.time()
            y_pred = model.predict(X).reshape(-1,1)
            scaler = MinMaxScaler()
            y_pred = scaler.fit_transform(y_pred)
            time_end = time.time()
            testtime = time_end - time_start
            mae = np.abs(density - model.density(X)).mean()
            roc_auc = roc_auc_score(y, y_pred)
            log_file_name = "NNDAD-S.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "NNDAD-S", iteration, dim, n, ratio, 
                    trainingtime, testtime, roc_auc, mae, n_estimators, lamda )
                f.writelines(logs)
            
          
                
                
                
        