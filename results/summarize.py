import os
import numpy as np 
import pandas as pd
import glob


log_file_dir = "./simulation_result"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,dim,anll,L2,mae,mse,time,best_params,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["dim", "method"],columns=["distribution"], values=[ "L2","mae"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./sorted_result/simulation_summary.xlsx")


