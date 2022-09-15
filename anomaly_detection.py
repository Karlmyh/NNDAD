'''
Preprocessing 
'''


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree
from scipy.stats import rankdata
from AWNN import AWNN
from KNN import KNN 
from AKDE import AKDE


import pandas as pd
import numpy as np
import glob
import os

data_file_dir = "./dataset/anomaly/"
method_seq = glob.glob("{}/*.csv".format(data_file_dir))
data_file_name_seq = [os.path.split(method)[1] for method in method_seq]
log_file_dir = "./anomaly_result/"


def awnn_predict_c(C):
    model_AWNN=AWNN(C=C).fit(X_train)
    scaler=MinMaxScaler()
    y_pred=scaler.fit_transform(model_AWNN.predict(X_train).reshape(-1,1))
    y_pred[y_pred==0]=1
    return roc_auc_score(y_train,y_pred)
    

def calculate_auc(X_train,y_train):
    
    roc_auc_vec=[]
    
    # IForest
    roc_auc=0
    for n_estimators in [100,300,500]:
        model_IF=IsolationForest(random_state=1,n_estimators=n_estimators).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_IF.decision_function(X_train).reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    roc_auc_vec.append(roc_auc)
    
    # LOF
    roc_auc=0
    for n_neighbors in [5*i for i in range(1,11)]:
        model_LOF=LocalOutlierFactor(n_neighbors=n_neighbors).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_LOF.negative_outlier_factor_.reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    roc_auc_vec.append(roc_auc)
    
    # OCSVM
    roc_auc=0
    for gamma in [1e-3,1e-2,1e-1,1,1e1]:
        model_OCSVM=OneClassSVM(gamma=gamma).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_OCSVM.decision_function(X_train).reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    roc_auc_vec.append(roc_auc)

    # KNN
    try:
        roc_auc=0
        tree_KNN=KDTree(X_train)
        for n_neighbors in [5*i for i in range(1,11)]:
            distance_vec,_=tree_KNN.query(X_train,n_neighbors+1)
            distance_vec=distance_vec[:,-1]
            scaler=MinMaxScaler()
            y_pred=scaler.fit_transform(-distance_vec.reshape(-1,1))
            roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
        roc_auc_vec.append(roc_auc)
    except:
        roc_auc_vec.append(0)

    # AWNN
    roc_auc=0
    try:
        for C in [i for i in np.logspace(-3,3,30)]:
            model_AWNN=AWNN(C=C).fit(X_train,max_neighbors=X_train.shape[0]-1)
            scaler=MinMaxScaler()
            y_pred=scaler.fit_transform(model_AWNN.predict(X_train).reshape(-1,1))
            roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))

        roc_auc_vec.append(roc_auc)
    except:
        roc_auc_vec.append(0)
    
    # AKDE
    roc_auc=0
    try:
        for C in [0.1,1,10]:
            for k in [1,3]:
                model_AKDE=AKDE(c=C,k=k).fit(X_train)
                scaler=MinMaxScaler()
                y_pred=scaler.fit_transform(model_AKDE.predict(X_train).reshape(-1,1))
                roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))

        roc_auc_vec.append(roc_auc)
    except:
        roc_auc_vec.append(0)
    
    # BKNN
    roc_auc=0
    try:
        for C in [i for i in np.logspace(-2,2,5)]:
            model_BKNN=KNN(C=C).fit(X_train,method="BKNN")
            scaler=MinMaxScaler()
            y_pred=scaler.fit_transform(model_BKNN.predict(X_train).reshape(-1,1))
            #y_pred[y_pred==0]=1
            roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))

        roc_auc_vec.append(roc_auc)
    except:
        roc_auc_vec.append(0)
        
    # TKNN
    roc_auc=0
    try:
        for threshold_num in [1,2,3]:
            for threshold_r in [0.01,0.1,1]:
                model_BKNN=KNN(threshold_num=threshold_num,threshold_r=threshold_r).fit(X_train,method="AKNN")
                scaler=MinMaxScaler()
                y_pred=scaler.fit_transform(model_BKNN.predict(X_train).reshape(-1,1))
                #y_pred[y_pred==0]=1
                roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))

        roc_auc_vec.append(roc_auc)
    except:
        roc_auc_vec.append(0)
    
    return roc_auc_vec


for data_file_name in data_file_name_seq:
    print(data_file_name)
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path)
    data = np.array(data)
    X_train=data[:,:-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    y_train=data[:,-1]
    y_train=1-2*y_train
    roc_auc_vec=calculate_auc(X_train,y_train)
    print(roc_auc_vec)
    log_file_name = "anomaly.csv"
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},".format(data_file_name.split(".")[0],
                                        roc_auc_vec[0],roc_auc_vec[1], roc_auc_vec[2],roc_auc_vec[3],
                                        roc_auc_vec[4],roc_auc_vec[5],roc_auc_vec[6],roc_auc_vec[7])
        f.writelines(logs)
        logs= "{},{},{},{},{},{},{}\n".format(9-rankdata(roc_auc_vec)[0],
        9-rankdata(roc_auc_vec)[1],9-rankdata(roc_auc_vec)[2],
        9-rankdata(roc_auc_vec)[3],9-rankdata(roc_auc_vec)[4],
        9-rankdata(roc_auc_vec)[5],9-rankdata(roc_auc_vec)[6],9-rankdata(roc_auc_vec)[7])
        f.writelines(logs)
        


