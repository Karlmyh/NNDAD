import os
import time
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import mat73
from NNDAD import NNDAD, BNNDAD
from distribution import TestDistribution


iter_times = 50
dim_vec = [1, 2, 5, 8]
n_vec = [1000 + 5000 * i for i in range(6)]
ratio_vec = [0.01,0.1]
log_file_dir = "./results/simulation/"


for iteration in range(iter_times):
    for dim, n, ratio in product(dim_vec, n_vec, ratio_vec):

        distribution = TestDistribution(index = 5, dim = dim).returnDistribution()
        X, density = distribution.generate(n)
        threshold = np.quantile(density, ratio)
        y = (density < threshold).astype(int)




#         scaler = MinMaxScaler()
#         X = scaler.fit_transform(X)
#         X = X[:,
#                           np.logical_not(
#                               np.isclose(X.min(axis=0), X.max(
#                                   axis=0)))]


#         # bagged nearest neighbor distance self-tuning
#         lamda_seq = [
#             0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
#             0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
#         ]
#         n_estimators_seq = [1, 5, 20, 50, 100]
#         tmp = 32
#         max_samples_seq = []
#         while tmp < X.shape[0]:
#             if tmp > 1024:
#                 break
#             max_samples_seq.append(tmp)
#             tmp *= 2
#         for n_estimators, max_samples in product(n_estimators_seq,
#                                                         max_samples_seq):
#             time_start = time.time()
#             model = BNNDAD(n_estimators=n_estimators,
#                            max_samples=max_samples,
#                            lamda_list=lamda_seq,
#                            random_state=101,
#                            n_jobs=n_jobs)
#             model.fit(X)
#             time_end = time.time()
#             trainingtime = time_end - time_start
#             time_start = time.time()
#             y_pred = model.predict(X)
#             scaler = MinMaxScaler()
#             y_pred = scaler.fit_transform(y_pred)
#             time_end = time.time()
#             testtime = time_end - time_start
#     #         mae = np.abs(density - )
#             roc_auc = roc_auc_score(y, y_pred)
#             log_file_name = "BNNDAD_selftuning.csv"
#             log_file_path = os.path.join(log_file_dir, log_file_name)
#             with open(log_file_path, "a") as f:
#                 logs = "{},{},{},{},{},{},{},{},{},{}\n".format(
#                     iteration, dim, n, ratio, n_estimators,
#                     max_samples, trainingtime, testtime, roc_auc, "")
#                 f.writelines(logs)
                
                
                
        # nearest neighbor distance self-tuning
        lamda_seq = [0.2 * i + 0.1 for i in range(25)]
        
        time_start = time.time()
        model = NNDAD(
                       lamda_list=lamda_seq,
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
        log_file_name = "NNDAD-S-ld-f.csv"
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                iteration, dim, n, ratio, "",
                "", trainingtime, testtime, roc_auc, mae, model.lamda )
            f.writelines(logs)

            
        # nearest neighbor distance best
        lamda_seq = [0.2 * i + 0.1 for i in range(25)]
        
        for lamda in lamda_seq:
            time_start = time.time()
            model = NNDAD(
                           lamda_list=[lamda],
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
            log_file_name = "NNDAD-B-ld-f.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    iteration, dim, n, ratio, "",
                    "", trainingtime, testtime, roc_auc, mae, model.lamda )
                f.writelines(logs)
         
        
        # bagged nearest neighbor distance best
        lamda_seq = [0.2 * i + 0.1 for i in range(25)]
        
        for lamda in lamda_seq:
            time_start = time.time()
            model = BNNDAD(
                       n_estimators=n_estimators,
                       max_samples=max_samples,
                       lamda_list=lamda_seq,
                       random_state=101,
                       n_jobs=n_jobs
                           lamda_list=[lamda],
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
            log_file_name = "NNDAD-B-ld-f.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    iteration, dim, n, ratio, "",
                    "", trainingtime, testtime, roc_auc, mae, model.lamda )
                f.writelines(logs)
                
        # bagged nearest neighbor distance self tuning
        lamda_seq = [0.2 * i + 0.1 for i in range(25)]
        
        time_start = time.time()
        model = BNNDAD(
                       n_estimators=n_estimators,
                       max_samples=max_samples,
                       lamda_list=lamda_seq,
                       random_state=101,
                       n_jobs=n_jobs
                           lamda_list=[lamda],
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
        log_file_name = "NNDAD-S-ld-f.csv"
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                iteration, dim, n, ratio, "",
                "", trainingtime, testtime, roc_auc, mae, model.lamda )
            f.writelines(logs)
            
          
                
                
                
        