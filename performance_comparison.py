import numpy as np
import os
import time
import scipy
import math


from AWNN import AWNN
from AKDE import AKDE
from synthetic_distributions import TestDistribution
from KNN import KNN
from sklearn.neighbors import KernelDensity

from sklearn.model_selection import GridSearchCV
#from scipy.stats import wilcoxon



def mae(x,y):
    return np.abs(x-y).mean()
def mse(x,y):
    return ((x-y)**2).mean()



dim_vec=[8]

distribution_index_vec=[4,7,8,9,12,13,17]
repeat_time=10
cv_criterion="MISE"

log_file_dir = "./simulation_result/"

for dim_iter,dim in enumerate(dim_vec):
    

    for distribution_iter,distribution_index in enumerate(distribution_index_vec):
        
        
        for iterate in range(repeat_time):
            
            
            
            np.random.seed(iterate)
            # generate distribution
           
            
            density=TestDistribution(distribution_index,dim).returnDistribution()
            n_test, n_train = 2000,1000
            X_train, pdf_X_train = density.generate(n_train)
            X_test, pdf_X_test = density.generate(n_test)
            
            #### score mae mse    method time C(parameter) iter ntrain ntest
          
            # AWNN
            time_start=time.time()
            parameters={"C":[i for i in np.logspace(-1.5,1.5,15)]}
            cv_model_AWNN=GridSearchCV(estimator=AWNN(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_AWNN.fit(X_train)
            time_end=time.time()
            log_file_name = "{}.csv".format("AWNN")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_AWNN=cv_model_AWNN.best_estimator_
            _=model_AWNN.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_AWNN.ANLL,-model_AWNN.MISE,mae(np.exp(model_AWNN.log_density),pdf_X_test),
                                              mse(np.exp(model_AWNN.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_AWNN.best_params_["C"],iterate,n_train,n_test)
                f.writelines(logs)
            
            
            
            # KNN
            time_start=time.time()
            parameters={"k":[int(i*n_train) for i in np.logspace(-2.5,np.log(2/3)/np.log(10),15)]}
            cv_model_KNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_KNN.fit(X_train,method="KNN")
            time_end=time.time()
            log_file_name = "{}.csv".format("KNN")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_KNN=cv_model_KNN.best_estimator_
            _=model_KNN.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_KNN.ANLL,-model_KNN.MISE,mae(np.exp(model_KNN.log_density),pdf_X_test),
                                              mse(np.exp(model_KNN.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_KNN.best_params_["k"],iterate,n_train,n_test)
                f.writelines(logs)
            
            # WKNN
            time_start=time.time()
            parameters={"k":[int(i*n_train) for i in np.logspace(-2.5,np.log(2/3)/np.log(10),15)]}
            
            cv_model_WKNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_WKNN.fit(X_train,method="WKNN")
            time_end=time.time()
            log_file_name = "{}.csv".format("WKNN")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_WKNN=cv_model_WKNN.best_estimator_
            _=model_WKNN.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_WKNN.ANLL,-model_WKNN.MISE,mae(np.exp(model_WKNN.log_density),pdf_X_test),
                                              mse(np.exp(model_WKNN.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_WKNN.best_params_["k"],iterate,n_train,n_test)
                f.writelines(logs)
            
            
            # AKNN
            time_start=time.time()
            parameters={"k":[int(i*n_train) for i in np.logspace(-2.5,np.log(2/3)/np.log(10),5)],
                        "threshold_r":[i for i in np.logspace(-3,0,3)],
                        "threshold_num":[1,2,3]}
            cv_model_AKNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_AKNN.fit(X_train,method="AKNN")
            time_end=time.time()
            log_file_name = "{}.csv".format("AKNN")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_AKNN=cv_model_AKNN.best_estimator_
            _=model_AKNN.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_AKNN.ANLL,-model_AKNN.MISE,mae(np.exp(model_AKNN.log_density),pdf_X_test),
                                              mse(np.exp(model_AKNN.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_AKNN.best_params_["k"],iterate,n_train,n_test)
                f.writelines(logs)
            
            
            # Kernel
            time_start=time.time()
            parameters={"bandwidth":[i for i in np.logspace(-1,1,15)]}
            cv_model_KDE=GridSearchCV(estimator=KernelDensity(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_KDE.fit(X_train)
            time_end=time.time()
            log_file_name = "{}.csv".format("KDE")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_KDE=cv_model_KDE.best_estimator_
            h=cv_model_KDE.best_params_["bandwidth"]
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              -cv_model_KDE.score(X_test)/n_test+1,-2*np.exp(model_KDE.score_samples(X_test)).mean()+np.exp(-scipy.spatial.distance.cdist(X_train,X_train)/4/h**2).sum()/h/n_train**2/2/math.pi,
                                              mae(np.exp(model_KDE.score_samples(X_test)),pdf_X_test),
                                              mse(np.exp(model_KDE.score_samples(X_test)),pdf_X_test),time_end-time_start,
                                              cv_model_KDE.best_params_["bandwidth"],iterate,n_train,n_test)
                f.writelines(logs)
            
            # AKDE
            time_start=time.time()
            parameters={"k":[int(i*n_train) for i in np.logspace(-2.5,np.log(2/3)/np.log(10),4)],
                        "c":[ c for c in np.logspace(-2,1,4)]}
            cv_model_AKDE=GridSearchCV(estimator=AKDE(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_AKDE.fit(X_train)
            time_end=time.time()
            log_file_name = "{}.csv".format("AKDE")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_AKDE=cv_model_AKDE.best_estimator_
            _=model_AKDE.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_AKDE.ANLL,-model_AKDE.MISE,mae(np.exp(model_AKDE.log_density),pdf_X_test),
                                              mse(np.exp(model_AKDE.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_AKDE.best_params_["k"],iterate,n_train,n_test)
                f.writelines(logs)
          
            # BKNN
            time_start=time.time()
            parameters={"C":[0.1,0.01,1,10,100]}
            cv_model_KNN=GridSearchCV(estimator=KNN(),param_grid=parameters,n_jobs=-1,cv=10)
            cv_model_KNN.fit(X_train,method="BKNN")
            time_end=time.time()
            log_file_name = "{}.csv".format("BKNN")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            model_KNN=cv_model_KNN.best_estimator_
            _=model_KNN.score(X_test)
            with open(log_file_path, "a") as f:
                logs= "{},{},{:.2e},{:.2e},{:.2e},{:.2e},{},{},{},{},{}\n".format(distribution_index,dim,
                                              model_KNN.ANLL,-model_KNN.MISE,mae(np.exp(model_KNN.log_density),pdf_X_test),
                                              mse(np.exp(model_KNN.log_density),pdf_X_test),time_end-time_start,
                                              cv_model_KNN.best_params_["C"],iterate,n_train,n_test)
                f.writelines(logs)
            
            
       