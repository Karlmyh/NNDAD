import os
import time
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import mat73

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree
from itertools import product



for data_file_name in [
#      "11_donors",
#     "12_fault",
#     "13_fraud",
#     "1_ALOI",
#     "33_skin",
#     "3_backdoor",
#     "5_campaign",
#     "8_celeba",
    "9_census",
    ]:
    data = np.load('./dataset/anomaly/{}.npz'.format(data_file_name), allow_pickle=True)
    
    X_train = data["X"]
    y_train = data["y"]
    log_file_dir = "./results/anomaly_realdata/"

    n_jobs = 20



    print(data_file_name, y_train.shape)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = X_train[:,
                      np.logical_not(
                          np.isclose(X_train.min(axis=0), X_train.max(
                              axis=0)))]
    log_file_name = "realdata_adbench.csv"
    log_file_path = os.path.join(log_file_dir, log_file_name)

#     # bagged nearest neighbor distance self-tuning
#     from NNDAD import BNNDAD
#     lamda_seq = [
#         0.0001, 0.0002, 0.0005, 0.001, 
#         0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
#         0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
#     ]
#     n_estimators_seq = [ 
#         1,
#                         5,
#                         20,
#                         50,
#                         100
#     ]
#     tmp = 32
#     max_samples_seq = []
#     while tmp < X_train.shape[0]:
#         if tmp > 2048:
#             break
#         max_samples_seq.append(tmp)
#         tmp *= 2
#     for n_estimators, max_samples, lamda in product(n_estimators_seq,
#                                                     max_samples_seq,
#                                                     lamda_seq):
#         time_start = time.time()
#         model = BNNDAD(n_estimators=n_estimators,
#                        max_samples=max_samples,
#                        lamda_list=[lamda],
#                        random_state=101,
#                        n_jobs=n_jobs)
#         model.fit(X_train)
#         time_end = time.time()
#         trainingtime = time_end - time_start
#         time_start = time.time()
#         y_pred = model.predict(X_train)
#         scaler = MinMaxScaler()
#         y_pred = scaler.fit_transform(y_pred)
#         time_end = time.time()
#         testtime = time_end - time_start

#         roc_auc = roc_auc_score(y_train, y_pred)
#         with open(log_file_path, "a") as f:
#             logs = "{},{},{},{},{},{},{},{}\n".format(
#                 data_file_name.split(".")[0], "BNNDAD", n_estimators,
#                 max_samples, lamda, trainingtime, testtime, roc_auc)
#             f.writelines(logs)

#     n_estimators = 1
#     max_samples = 1.0
#     for lamda in lamda_seq:
#         time_start = time.time()
#         model = BNNDAD(n_estimators=n_estimators,
#                        max_samples=max_samples,
#                        lamda_list=[lamda],
#                        random_state=101,
#                        n_jobs=n_jobs)
#         model.fit(X_train)
#         time_end = time.time()
#         trainingtime = time_end - time_start
#         time_start = time.time()
#         y_pred = model.predict(X_train)
#         scaler = MinMaxScaler()
#         y_pred = scaler.fit_transform(y_pred)
#         time_end = time.time()
#         testtime = time_end - time_start

#         roc_auc = roc_auc_score(y_train, y_pred)
#         with open(log_file_path, "a") as f:
#             logs = "{},{},{},{},{},{},{},{}\n".format(
#                 data_file_name.split(".")[0], "NNDAD", n_estimators,
#                 max_samples, lamda, trainingtime, testtime, roc_auc)
#             f.writelines(logs)


#     # IForest
#     roc_auc=0
#     for n_estimators in [100,300,500]:
#         time_start = time.time()
#         model_IF=IsolationForest(random_state=1,n_estimators=n_estimators).fit(X_train,y_train)
#         time_end = time.time()
#         trainingtime = time_end - time_start
#         time_start = time.time()
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(model_IF.decision_function(X_train).reshape(-1,1))
#         roc_auc=max(roc_auc,roc_auc_score(y_train, - y_pred))
#         time_end = time.time()
#         testtime = time_end - time_start
#         with open(log_file_path, "a") as f:
#             logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"IF", n_estimators, "", "",trainingtime, testtime, roc_auc)
#             f.writelines(logs)

    # LOF
    roc_auc=0
    for n_neighbors in [50*i for i in range(1,21)]:
        time_start = time.time()
        model_LOF=LocalOutlierFactor(n_neighbors = n_neighbors).fit(X_train,y_train)
        time_end = time.time()
        trainingtime = time_end - time_start
        time_start = time.time()
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_LOF.negative_outlier_factor_.reshape(-1,1))
        time_end = time.time()
        testtime = time_end - time_start
        roc_auc = roc_auc_score(y_train, - y_pred)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"LOF", n_neighbors, "", "",trainingtime, testtime, roc_auc)
            f.writelines(logs)

#     # OCSVM
#     roc_auc=0
#     for gamma in [1e-3,1e-2,1e-1,1,1e1]:
#         time_start = time.time()
#         model_OCSVM=OneClassSVM(gamma=gamma).fit(X_train,y_train)
#         time_end = time.time()
#         trainingtime = time_end - time_start
#         time_start = time.time()
#         scaler=MinMaxScaler()
#         y_pred=scaler.fit_transform(model_OCSVM.decision_function(X_train).reshape(-1,1))
#         time_end = time.time()
#         testtime = time_end - time_start
#         roc_auc = roc_auc_score(y_train, - y_pred)
#         with open(log_file_path, "a") as f:
#             logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"OCSVM", gamma, "", "", trainingtime, testtime, roc_auc)
#             f.writelines(logs)

#     # KNN
#     roc_auc=0
#     time_start = time.time()
#     tree_KNN=KDTree(X_train)
#     time_end = time.time()
#     trainingtime = time_end - time_start
#     for n_neighbors in [50*i for i in range(1,21)]:
#         time_start = time.time()
#         distance_vec,_=tree_KNN.query(X_train,n_neighbors+1)
#         distance_vec=distance_vec[:,-1]
#         scaler=MinMaxScaler()
#         time_end = time.time()
#         testtime = time_end - time_start
#         roc_auc = roc_auc_score(y_train,distance_vec)
#         with open(log_file_path, "a") as f:
#             logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"KNN", n_neighbors, "", "",trainingtime, testtime, roc_auc)
#             f.writelines(logs)



#     # pidforest 
    
    
#     from comparison.scripts import Forest
#     roc_auc=0
#     n_samples_seq = [50, 100, 200, 400,600,800,1000]
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
#             time_start = time.time()
#             forest = Forest(**kwargs)
            
#             forest.fit(np.transpose(X_train))
#             time_end = time.time()
#             trainingtime = time_end - time_start
#             time_start = time.time()
#             indices, outliers, scores, pst, our_scores = forest.predict(
#                 np.transpose(X_train), err=0.1, pct=50)
#             time_end = time.time()
#             testtime = time_end - time_start
#             roc_auc = roc_auc_score(y_train, -our_scores)
            
#             with open(log_file_path, "a") as f:
#                 logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"PID", n_samples, max_buckets,  "",trainingtime, testtime, roc_auc)
#                 f.writelines(logs)
#             # 2.3.4 save results
#         except Exception as e:
#             print(e)

        

   
            
            
#     # DTM 
    
#     from comparison.DTM import DTM
#     roc_auc=0
    
#     n_neighbors_seq = [10,50, 100, 200, 400, 600, 800, 1000]
    
#     for n_neighbors in n_neighbors_seq:
#         try:
#             if n_neighbors > X_train.shape[0]:
#                 continue
#             time_start = time.time()
#             clf_DTM = DTM(n_neighbors=n_neighbors, n_jobs = 10)
#             time_end = time.time()
#             trainingtime = time_end - time_start
#             time_start = time.time()
#             y_score = clf_DTM.fit_predict(X_train)
            
#             time_end = time.time()
#             testtime = time_end - time_start
            
#             roc_auc =  roc_auc_score(y_train, -y_score)
#             with open(log_file_path, "a") as f:
#                 logs= "{},{},{},{},{},{},{},{}\n".format(data_file_name.split(".")[0],"DTM", n_neighbors, "",  "",trainingtime, testtime, roc_auc)
#                 f.writelines(logs)
#             # 2.3.4 save results
#         except Exception as e:
#             print(e)

    