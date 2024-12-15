import os
import time
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from NNDAD import WNNDAD, NNDAD
from distribution import TestDistribution

# repeat times
iter_times = 20
# dimensions
dim_vec = [1]
# number of samples
n_vec = [1000]
# ratio of anomalies
k_vec = [ k for k in range(2, 500, 10)]
# potential bagging vec
alpha_vec = [10, 5, 2, 1, 0.5, 0.2 , 0.1]
# process number
n_jobs = 20


log_file_dir = "./results/simulation/"


for iteration in range(iter_times):
    for dim, n in product(dim_vec, n_vec):

        distribution = TestDistribution(index = 2, dim = dim).returnDistribution()
        X, density = distribution.generate(n)
        X_test, density_test = distribution.generate(10000)
        threshold = np.quantile(density, 0.05)
        y = (density < threshold).astype(int)


        time_start = time.time()
        model = NNDAD()
        model.fit(X)
        time_end = time.time()
        trainingtime = time_end - time_start
        time_start = time.time()
        y_pred = model.predict(X).reshape(-1,1)
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(y_pred)
        time_end = time.time()
        testtime = time_end - time_start
        mae = np.abs(density_test - model.density(X_test)).mean()
        roc_auc = roc_auc_score(y, y_pred)
        log_file_name = "weights.csv"
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                "NNDAD", iteration, dim, n, 0.05, 0, 0,
                trainingtime, testtime, roc_auc, mae )
            f.writelines(logs)


        # # bagged nearest neighbor distance self tuning
        for alpha, k  in product(alpha_vec, k_vec):
            time_start = time.time()
            model = WNNDAD(
                          alpha = alpha,
                            k = k,
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
            mae = np.abs(density_test - model.density(X_test)).mean()
            roc_auc = roc_auc_score(y, y_pred)
            log_file_name = "weights.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "WNNDAD", iteration, dim, n, 0.05, alpha, k,
                    trainingtime, testtime, roc_auc, mae )
                f.writelines(logs)

            time_start = time.time()
            model = WNNDAD(use_knn=True,
                           k = k,)
            model.fit(X)
            time_end = time.time()
            trainingtime = time_end - time_start
            time_start = time.time()
            y_pred = model.predict(X).reshape(-1,1)
            scaler = MinMaxScaler()
            y_pred = scaler.fit_transform(y_pred)
            time_end = time.time()
            testtime = time_end - time_start
            mae = np.abs(density_test - model.density(X_test)).mean()
            roc_auc = roc_auc_score(y, y_pred)
            log_file_name = "weights.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "kNN", iteration, dim, n, 0.05, alpha, k,
                    trainingtime, testtime, roc_auc, mae )
                f.writelines(logs)

        


            
          
                
                
                
        