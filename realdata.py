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
from NNDAD import NNDAD, NNDADDIST
from scipy.io import loadmat
import mat73

data_file_dir = "./dataset/anomaly_raw/"
method_seq = glob.glob("{}/*.mat".format(data_file_dir))
data_file_name_seq = [os.path.split(method)[1] for method in method_seq]
log_file_dir = "./results/anomaly_realdata/"

  
data_file_name_seq = [ 
#     'satellite.mat',
#  'http.mat',
 'wine.mat',
#  'smtp.mat',
#  'vowels.mat',
#  'ionosphere.mat',
#  'arrhythmia.mat',
#  'glass.mat',
#  'thyroid.mat',
#  'cardio.mat',
#  'shuttle.mat',
#  'breastw.mat',
#  'letter.mat',
#  'optdigits.mat',
#  'vertebral.mat',
#  'lympho.mat',
#  'mammography.mat',
#  'musk.mat',
#  'wbc.mat',
#  'annthyroid.mat',
#  'cover.mat',
#  'satimage-2.mat',
#  'mnist.mat',
#  'pima.mat',
#  'pendigits.mat'
                     ]

for data_file_name in data_file_name_seq:
    print(data_file_name)
    data_file_path = os.path.join(data_file_dir, data_file_name)
    if data_file_name in ["smtp.mat","http.mat"]:
        data = mat73.loadmat(data_file_path)
    else:
        data = loadmat(data_file_path)
    X_train = data["X"]
    y_train = data["y"].reshape(-1)
    print(y_train.shape)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
#         y_train=1-2*y_train
    X_train = X_train[:, np.logical_not(np.isclose(X_train.min(axis=0), X_train.max(axis=0)))]
#     y_train=1-2*y_train
    log_file_name = "realdata.csv"
    log_file_path = os.path.join(log_file_dir, log_file_name)
    
    # nearest neighbor distance self-tuning
    if X_train.shape[0] < 20000:
        model_NNDAD = NNDAD(lamda_list = [0.0001,0.001, 0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]).fit(X_train)
    else:
        model_NNDAD = NNDAD(max_samples_ratio = 0.5 , lamda_list = [0.0001,0.001, 0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]).fit(X_train)
    print(X_train.shape[0], (model_NNDAD.weights > 0).sum())
    
    weight_file_path = os.path.join(log_file_dir, "weight_{}.npy".format(data_file_name.split(".")[0]))
    np.save(weight_file_path, model_NNDAD.weights) 
    scaler = MinMaxScaler()
    y_pred = scaler.fit_transform(  model_NNDAD.predict(X_train).reshape(-1,1))
    roc_auc = roc_auc_score(y_train, y_pred)
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"NNDAD", roc_auc)
        f.writelines(logs)
       
#     # weighted nearest neighbor distance distributed
#     roc_auc = 0
#     print()
#     for lamda in [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]:
#         if X_train.shape[0] > 30000:
#             for distributed_fold in [20,40]:
#                 model_WNN = NNDADDIST(lamda_list = [lamda], distributed_fold = distributed_fold).fit(X_train)
#                 print(X_train.shape[0], X_train.shape[1], (model_WNN.weights > 0).sum())
#                 scaler = MinMaxScaler()
#                 y_pred = scaler.fit_transform(  model_WNN.predict(X_train).reshape(-1,1))
#                 roc_auc = max(roc_auc, roc_auc_score(y_train,y_pred))
#         else:
#             for distributed_fold in [2,5,10,20,40]:
#                 model_WNN = NNDADDIST(lamda_list = [lamda], distributed_fold = distributed_fold).fit(X_train)
#                 print(X_train.shape[0], X_train.shape[1], (model_WNN.weights > 0).sum())
#                 scaler = MinMaxScaler()
#                 y_pred = scaler.fit_transform(  model_WNN.predict(X_train).reshape(-1,1))
#                 roc_auc = max(roc_auc, roc_auc_score(y_train,y_pred))
#                 print(roc_auc, distributed_fold)
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"WNNDIST2", roc_auc)
#         f.writelines(logs)

#     # nearest neighbor distance self-tuning DIST
#     roc_auc = 0
#     if X_train.shape[0] > 20000:
#         for distributed_fold in [20,40]:
#             model_NNDAD = NNDADDIST(lamda_list = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10], distributed_fold = distributed_fold ).fit(X_train)
#             print(X_train.shape[0], (model_NNDAD.weights > 0).sum())
#             scaler = MinMaxScaler()
#             y_pred = scaler.fit_transform(  model_NNDAD.predict(X_train).reshape(-1,1))
#             roc_auc = max(roc_auc, roc_auc_score(y_train,y_pred))
#     else:
#         for distributed_fold in [2,5,10,20,40]:
#             model_NNDAD = NNDADDIST(lamda_list = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10], distributed_fold = 40).fit(X_train)
#             print(X_train.shape[0], (model_NNDAD.weights > 0).sum())
#             scaler = MinMaxScaler()
#             y_pred = scaler.fit_transform(  model_NNDAD.predict(X_train).reshape(-1,1))
#             roc_auc = max(roc_auc, roc_auc_score(y_train,y_pred))
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"NNDADDIST2", roc_auc)
#         f.writelines(logs)
  
    
    
    # weighted nearest neighbor distance
    roc_auc = 0
    for lamda in [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]:
        model_WNN = NNDAD( lamda_list = [lamda], max_samples_ratio = 1).fit(X_train)
        print(X_train.shape[0], (model_WNN.weights > 0).sum())
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(  model_WNN.predict(X_train).reshape(-1,1))
        roc_auc = max(roc_auc, roc_auc_score(y_train,y_pred))
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"WNN", roc_auc)
        f.writelines(logs)

#     # IForest
#     roc_auc=0
#     for n_estimators in [100,300,500]:
#         model_IF=IsolationForest(random_state=1,n_estimators=n_estimators).fit(X_train,y_train)
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(model_IF.decision_function(X_train).reshape(-1,1))
#         roc_auc=max(roc_auc,roc_auc_score(y_train, - y_pred))
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"IF", roc_auc)
#         f.writelines(logs)

#     # LOF
#     roc_auc=0
#     for n_neighbors in [50*i for i in range(1,21)]:
#         model_LOF=LocalOutlierFactor(n_neighbors = n_neighbors).fit(X_train,y_train)
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(model_LOF.negative_outlier_factor_.reshape(-1,1))
#         roc_auc=max(roc_auc,roc_auc_score(y_train, - y_pred))
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"LOF", roc_auc)
#         f.writelines(logs)
        
#     # OCSVM
#     roc_auc=0
#     for gamma in [1e-3,1e-2,1e-1,1,1e1]:
#         model_OCSVM=OneClassSVM(gamma=gamma).fit(X_train,y_train)
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(model_OCSVM.decision_function(X_train).reshape(-1,1))
#         roc_auc=max(roc_auc,roc_auc_score(y_train, - y_pred))
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"OCSVM", roc_auc)
#         f.writelines(logs)
        
#     # KNN
#     roc_auc=0
#     tree_KNN=KDTree(X_train)
#     for n_neighbors in [50*i for i in range(1,21)]:
#         distance_vec,_=tree_KNN.query(X_train,n_neighbors+1)
#         distance_vec=distance_vec[:,-1]
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(distance_vec.reshape(-1,1))
#         roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
#         print(n_neighbors, roc_auc)
#     with open(log_file_path, "a") as f:
#         logs= "{},{},{}\n".format(data_file_name.split(".")[0],"KNN", roc_auc)
#         f.writelines(logs)


    
   
    
  
    
  


