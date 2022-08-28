import os
import time
import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import numpy as np
import os

import scipy
import math

from AWNN import AWNN
from sklearn.neighbors import KernelDensity
from AKDE import AKDE
from KNN import KNN

from sklearn.model_selection import GridSearchCV
from scipy.stats import wilcoxon

data_file_dir = "./dataset/density_dataset"
#data_file_name_seq = ['ionosphere.csv','adult.csv','abalone.csv', 'australian.csv', 'breast-cancer.csv', 'credit.csv', 'parkinsons.csv', 'winequality-red.csv', 'winequality-white.csv', 'winequality.csv']
data_file_name_seq = ['abalone.csv', 'australian.csv', 'breast-cancer.csv', 'credit.csv', 'parkinsons.csv', 'winequality-red.csv', 'winequality-white.csv', 'winequality.csv']
#data_file_name_seq=["lympho.csv","cardio.csv", "thyroid.csv","vowels.csv", "glass.csv", "musk.csv","letter.csv", "pima.csv", "satellite.csv", "pendigits.csv", "yeast.csv", "heart.csv"]
#data_file_name_seq=['ionosphere.csv','adult.csv', 'winequality.csv']

log_file_dir = "./realdata_result/"

def anll(pdf):
    return -np.log(pdf).mean()


def calculate_score(X_train, X_test):

    n_train=X_train.shape[0]

    time_vec=np.zeros(6)
    L2_vec=np.array([])
    ANLL_vec=np.array([])
    params_vec=[]

    
    # AWNN
    time_start=time.time()
    parameters={"C":[i for i in np.logspace(2,5,10)],"cut_off":[1,5]}
    cv_model_AWNN=GridSearchCV(estimator=AWNN(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_AWNN.fit(X_train)
    model_AWNN=cv_model_AWNN.best_estimator_
    
    L2_vec=np.append(L2_vec, -model_AWNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_AWNN.ANLL)
    time_end=time.time()
    time_vec[0]+=time_end-time_start
    params_vec.append(cv_model_AWNN.best_params_["C"])
    
    # KNN
    time_start=time.time()
    parameters={"k":[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),15)]}
    cv_model_KNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_KNN.fit(X_train,method="KNN")
    model_KNN=cv_model_KNN.best_estimator_
    
    L2_vec=np.append(L2_vec, -model_KNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_KNN.ANLL)
    time_end=time.time()
    time_vec[1]+=time_end-time_start
    params_vec.append(cv_model_KNN.best_params_["k"])
    
    # WKNN
    time_start=time.time()
    parameters={"k":[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),15)]}
    cv_model_WKNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_WKNN.fit(X_train,method="WKNN")
    model_WKNN=cv_model_WKNN.best_estimator_
    
    L2_vec=np.append(L2_vec, -model_WKNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_WKNN.ANLL)
    time_end=time.time()
    time_vec[2]+=time_end-time_start
    params_vec.append(cv_model_WKNN.best_params_["k"])
    
    # AKNN
    time_start=time.time()
    parameters={"k":[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),5)],
                "threshold_r":[0.01,0.1,1],
                "threshold_num":[1,2,3]}
    cv_model_AKNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_AKNN.fit(X_train,method="AKNN")
    model_AKNN=cv_model_AKNN.best_estimator_
    
    L2_vec=np.append(L2_vec, -model_AKNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_AKNN.ANLL)
    time_end=time.time()
    time_vec[3]+=time_end-time_start
    params_vec.append(cv_model_AKNN.best_params_["k"])
    
    
    # KDE
    time_start=time.time()
    parameters={"bandwidth":[i for i in np.logspace(-1,1,15)]}
    cv_model_KDE=GridSearchCV(estimator=KernelDensity(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_KDE.fit(X_train)
    model_KDE=cv_model_KDE.best_estimator_
    h=cv_model_KDE.best_params_["bandwidth"]
    
    L2_vec=np.append(L2_vec, -2*np.exp(model_KDE.score_samples(X_test)).mean()+np.exp(-scipy.spatial.distance.cdist(X_train,X_train)/4/h**2).sum()/h/n_train**2/2/math.pi)
    ANLL_vec=np.append(ANLL_vec, -model_KDE.score_samples(X_test).mean())
    time_end=time.time()
    time_vec[4]+=time_end-time_start
    params_vec.append(cv_model_KDE.best_params_["bandwidth"])
    
    # AKDE
    time_start=time.time()
    parameters={"k":[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),5)],
                "c":[ c for c in np.logspace(-2,1,3)]}
    cv_model_AKDE=GridSearchCV(estimator=AKDE(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_AKDE.fit(X_train)
    model_AKDE=cv_model_AKDE.best_estimator_
    
    L2_vec=np.append(L2_vec, -model_AKDE.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_AKDE.ANLL)
    time_end=time.time()
    time_vec[5]+=time_end-time_start
    params_vec.append(cv_model_AKDE.best_params_["k"])
    
    return L2_vec,ANLL_vec,time_vec,params_vec


def calculate_score_largedata(X_train, X_test):

    n_train=X_train.shape[0]

    time_vec=np.zeros(5)
    L2_vec=np.array([])
    ANLL_vec=np.array([])
    params_vec=[]

    
    # AWNN
    time_start=time.time()
    L2_valid = []
    C_vec=[i for i in np.logspace(2,5,5)]
    for C in C_vec:
        model=AWNN(C=C)
        model.fit(X_train)
        L2_valid.append(model.compute_MISE(X_train))
    idx=np.array(L2_valid).argmin()
    best_C=C_vec[idx]
    model_AWNN=AWNN(C=best_C)
    model_AWNN.fit(X_train)
    L2_vec=np.append(L2_vec, -model_AWNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_AWNN.ANLL)
    time_end=time.time()
    time_vec[0]+=time_end-time_start
    params_vec.append(best_C)
    
    # KNN
    time_start=time.time()
    L2_valid = []
    k_vec=[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),5)]
    for k in k_vec:
        model=KNN(k=k)
        model.fit(X_train,method="KNN")
        L2_valid.append(model.compute_MISE(X_train))
    idx=np.array(L2_valid).argmin()
    best_k=k_vec[idx]
    model_KNN=KNN(k=best_k)
    model_KNN.fit(X_train)
    L2_vec=np.append(L2_vec, -model_KNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_KNN.ANLL)
    time_end=time.time()
    time_vec[1]+=time_end-time_start
    params_vec.append(best_k)
    
    # WKNN
    time_start=time.time()
    L2_valid = []
    k_vec=[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),5)]
    for k in k_vec:
        model=KNN(k=k)
        model.fit(X_train,method="WKNN")
        L2_valid.append(model.compute_MISE(X_train))
    idx=np.array(L2_valid).argmin()
    best_k=k_vec[idx]
    model_WKNN=KNN(k=best_k)
    model_WKNN.fit(X_train)
    L2_vec=np.append(L2_vec, -model_WKNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_WKNN.ANLL)
    time_end=time.time()
    time_vec[2]+=time_end-time_start
    params_vec.append(best_k)
    
    # AKNN
    time_start=time.time()
    L2_valid = []
    k_vec=[int(i*n_train) for i in np.logspace(np.log(2/n_train)/np.log(10),np.log(2/3)/np.log(10),5)]
    k_vec=np.repeat(k_vec,4)
    threshold_r_vec=[0.01,0.1,0.01,0.1]
    threshold_r_vec=np.tile(threshold_r_vec,5)
    threshold_num_vec=[1,1,2,2]
    threshold_num_vec=np.tile(threshold_num_vec,5)
    for k,threshold_r,threshold_num in zip(k_vec,threshold_r_vec,threshold_num_vec):
        model=KNN(k=k,threshold_r=threshold_r,threshold_num=threshold_num)
        model.fit(X_train,method="AKNN")
        L2_valid.append(model.compute_MISE(X_train))
    idx=np.array(L2_valid).argmin()
    best_k=k_vec[idx]
    best_threshold_r=threshold_r_vec[idx]
    best_threshold_num=threshold_num_vec[idx]
    model_AKNN=KNN(k=best_k, threshold_r=best_threshold_r, threshold_num=best_threshold_num)
    model_AKNN.fit(X_train)
    L2_vec=np.append(L2_vec, -model_AKNN.score(X_test))
    ANLL_vec=np.append(ANLL_vec, model_AKNN.ANLL)
    time_end=time.time()
    time_vec[3]+=time_end-time_start
    params_vec.append(best_k)
    
    # KDE
    time_start=time.time()
    parameters={"bandwidth":[i for i in np.logspace(-1,1,5)]}
    cv_model_KDE=GridSearchCV(estimator=KernelDensity(),param_grid=parameters,n_jobs=-1,cv=10)
    cv_model_KDE.fit(X_train)
    model_KDE=cv_model_KDE.best_estimator_
    h=cv_model_KDE.best_params_["bandwidth"]
    
    L2_vec=np.append(L2_vec, -2*np.exp(model_KDE.score_samples(X_test)).mean()+np.exp(-scipy.spatial.distance.cdist(X_train,X_train)/4/h**2).sum()/h/n_train**2/2/math.pi)
    ANLL_vec=np.append(ANLL_vec, -model_KDE.score_samples(X_test).mean())
    time_end=time.time()
    time_vec[4]+=time_end-time_start
    params_vec.append(cv_model_KDE.best_params_["bandwidth"])
    
    
   
   
    
    return L2_vec,ANLL_vec,time_vec,params_vec


def function_for_bknn(X_train,X_test):
    
    n_train=X_train.shape[0]

    
    time_start=time.time()
    L2_valid = []
    C_vec=[0.01,0.1,1,10]
    for C in C_vec:
        model=KNN(C=C)
        model.fit(X_train,method="BKNN")
        L2_valid.append(model.compute_MISE(X_train))
    idx=np.array(L2_valid).argmin()
    best_C=C_vec[idx]
    model_BKNN=KNN(C=best_C)
    model_BKNN.fit(X_train)
    L2=model_BKNN.score(X_test)
    time_end=time.time()
    
    
   
    
    return -L2, model_BKNN.ANLL,time_end-time_start,best_C

for data_file_name in data_file_name_seq:
    # load dataset
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path)
    data = np.array(data)
    # dataset status
    data_name = os.path.splitext(data_file_name)[0]
    num_samples, num_features = data.shape
    # transformation
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    dim_seq = np.array([0.3, 0.5, 0.7]) * num_features
    dim_seq = np.array(np.round(dim_seq), dtype=np.int32)
    #
    repeat_times = 5
    for dim in dim_seq:

        
        for i in range(repeat_times):

            # pca
            transformer = PCA(n_components=dim)
            transformed_data = transformer.fit_transform(data)
            train_X, test_X = train_test_split(transformed_data, train_size=0.7, test_size=0.3)
            # estimation
            L2,ANLL,time_cost,params= function_for_bknn(train_X, test_X)
            log_file_name = "realdata_BKNN.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL,L2,time_cost,params,i)
                f.writelines(logs)
            
            if train_X.shape[0]<100:
                L2_vec,ANLL_vec,time_vec,params_vec= calculate_score(train_X, test_X)
            else:
                L2_vec,ANLL_vec,time_vec,params_vec= calculate_score_largedata(train_X, test_X)
            
            log_file_name = "realdata_AWNN.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL_vec[0],L2_vec[0],time_vec[0],params_vec[0],i)
                f.writelines(logs)
                
            log_file_name = "realdata_KNN.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL_vec[1],L2_vec[1],time_vec[1],params_vec[1],i)
                f.writelines(logs)
                
            log_file_name = "realdata_WKNN.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL_vec[2],L2_vec[2],time_vec[2],params_vec[2],i)
                f.writelines(logs)
                
                
            log_file_name = "realdata_AKNN.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL_vec[3],L2_vec[3],time_vec[3],params_vec[3],i)
                f.writelines(logs)
                
                
            log_file_name = "realdata_KDE.csv"
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{}\n".format(data_file_name.split(".")[0],dim,
                                              ANLL_vec[4],L2_vec[4],time_vec[4],params_vec[4],i)
                f.writelines(logs)
                
                
         
        
            