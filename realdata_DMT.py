import numpy as np
import os
import glob
from time import time
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree
from scipy.stats import rankdata
from comparison.DTM import DTM
from comparison.scripts import Forest
from scipy.io import loadmat
from itertools import product
import mat73


data_file_dir = "./dataset/anomaly_raw/"
method_seq = glob.glob("{}/*.mat".format(data_file_dir))
data_file_name_seq = [os.path.split(method)[1] for method in method_seq]
log_file_dir = "./results/anomaly_realdata/"

  
for data_file_name in data_file_name_seq:
    print(data_file_name)
    
    try:
        data_file_path = os.path.join(data_file_dir, data_file_name)
        if data_file_name in ["http.mat"]:
            data = mat73.loadmat(data_file_path)
        else:
            data = loadmat(data_file_path)
        X_train = data["X"]
        y_train = data["y"].reshape(-1)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
#         y_train=1-2*y_train
        X_train = X_train[:, np.logical_not(np.isclose(X_train.min(axis=0), X_train.max(axis=0)))]
        log_file_name = "realdata.csv"
        log_file_path = os.path.join(log_file_dir, log_file_name)
    except Exception as e:
        print(e)

    
    # pidforest 
    
#     roc_auc=0
#     n_samples_seq = [50, 100, 200, 400]
#     max_buckets_seq = [3, 4, 5, 6]
#     for n_samples, max_buckets in product(n_samples_seq, max_buckets_seq):
#         try:
#             if n_samples > X_train.shape[0]:
#                 continue

#             max_depth = 10      
#             n_trees = 50
#             kwargs = {
#                 'max_depth': max_depth,
#                 'n_trees': n_trees,
#                 'max_samples': n_samples,
#                 'max_buckets': max_buckets,
#                 'epsilon': 0.1,
#                 'sample_axis': 1,
#                 'threshold': 0
#             }
#             forest = Forest(**kwargs)
#             forest.fit(np.transpose(X_train))
#             indices, outliers, scores, pst, our_scores = forest.predict(
#                 np.transpose(X_train), err=0.1, pct=50)
#             roc_auc = max(roc_auc, roc_auc_score(y_train, -our_scores))
#             # 2.3.4 save results
#         except Exception as e:
#             print(e)

#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"pidforest", roc_auc)
#         f.writelines(logs)

   
            
            
    # DTM 
    roc_auc=0
    
    n_neighbors_seq = [10,50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    
    for n_neighbors in n_neighbors_seq:
        try:
            if n_neighbors > X_train.shape[0]:
                continue
            clf_DTM = DTM(n_neighbors=n_neighbors, n_jobs = 20)
            y_score = clf_DTM.fit_predict(X_train)

            
            roc_auc = max(roc_auc, roc_auc_score(y_train, -y_score))
            # 2.3.4 save results
        except Exception as e:
            print(e)

    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"DTM", roc_auc)
        f.writelines(logs)
  

    
   
    
  
    
  


