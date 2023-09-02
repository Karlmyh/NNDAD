import os
import time
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import mat73
from NNDAD import BNNDAD

for data_file_name in [
     "11_donors",
    "12_fault",
    "13_fraud",
    "1_ALOI",
    "33_skin",
    "3_backdoor",
    "5_campaign",
    "8_celeba",
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
    
    # bagged nearest neighbor distance self-tuning
    lamda_seq = [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
    ]
    n_estimators_seq = [1, 5, 20, 50, 100]
    tmp = 32
    max_samples_seq = []
    while tmp < X_train.shape[0]:
        if tmp > 1024:
            break
        max_samples_seq.append(tmp)
        tmp *= 2
    for n_estimators, max_samples in product(n_estimators_seq,
                                                    max_samples_seq):
        time_start = time.time()
        model = BNNDAD(n_estimators=n_estimators,
                       max_samples=max_samples,
                       lamda_list=lamda_seq,
                       random_state=101,
                       n_jobs=n_jobs)
        model.fit(X_train)
        time_end = time.time()
        trainingtime = time_end - time_start
        time_start = time.time()
        y_pred = model.predict(X_train)
        scaler = MinMaxScaler()
        y_pred = scaler.fit_transform(y_pred)
        time_end = time.time()
        testtime = time_end - time_start

        roc_auc = roc_auc_score(y_train, y_pred)
        with open(log_file_path, "a") as f:
            logs = "{},{},{},{},{},{},{},{}\n".format(
                data_file_name.split(".")[0], "BNNDAD_selftuning", n_estimators,
                max_samples,  "", trainingtime, testtime, roc_auc)
            f.writelines(logs)
