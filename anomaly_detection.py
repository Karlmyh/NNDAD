


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree
from scipy.stats import rankdata
from NNDAD import NNDAD


data_file_dir = "./dataset/anomaly/"
method_seq = glob.glob("{}/*.csv".format(data_file_dir))
data_file_name_seq = [os.path.split(method)[1] for method in method_seq]
log_file_dir = "./results/anomaly_realdata/"

  
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
    log_file_name = "realdata.csv"
    log_file_path = os.path.join(log_file_dir, log_file_name)
    

    model_NNDAD = NNDAD( lamda_list = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]).fit(X_train)
    scaler = MinMaxScaler()
    y_pred = scaler.fit_transform( - model_NNDAD.predict(X_train).reshape(-1,1))
    roc_auc = roc_auc_score(y_train, y_pred)
    
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"NNDAD", roc_auc)
        f.writelines(logs)
       

    # IForest
    roc_auc=0
    for n_estimators in [100,300,500]:
        model_IF=IsolationForest(random_state=1,n_estimators=n_estimators).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_IF.decision_function(X_train).reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"IF", roc_auc)
        f.writelines(logs)

    # LOF
    roc_auc=0
    for n_neighbors in [5*i for i in range(1,11)]:
        model_LOF=LocalOutlierFactor(n_neighbors=n_neighbors).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_LOF.negative_outlier_factor_.reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"LOF", roc_auc)
        f.writelines(logs)
        
    # OCSVM
    roc_auc=0
    for gamma in [1e-3,1e-2,1e-1,1,1e1]:
        model_OCSVM=OneClassSVM(gamma=gamma).fit(X_train,y_train)
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(model_OCSVM.decision_function(X_train).reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"OCSVM", roc_auc)
        f.writelines(logs)
        
    # KNN
    roc_auc=0
    tree_KNN=KDTree(X_train)
    for n_neighbors in [5*i for i in range(1,11)]:
        distance_vec,_=tree_KNN.query(X_train,n_neighbors+1)
        distance_vec=distance_vec[:,-1]
        scaler=MinMaxScaler()
        y_pred=scaler.fit_transform(-distance_vec.reshape(-1,1))
        roc_auc=max(roc_auc,roc_auc_score(y_train,y_pred))
    with open(log_file_path, "a") as f:
        logs= "{},{},{}\n".format(data_file_name.split(".")[0],"KNN", roc_auc)
        f.writelines(logs)


    
   
    
  
    
  


