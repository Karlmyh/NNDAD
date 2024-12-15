import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


log_file_dir = "./results/simulation/weights.csv"

log = pd.read_csv(log_file_dir, header=None)
log.columns = "method,iter,dim,n,ratio,alpha,k,trainingtime,testtime,roc_auc,mae".split(',')
mean_log = log.groupby(by=["method", "alpha", "k", "dim", "n", "ratio"]).agg('mean').reset_index()

print(mean_log)

for alpha in [10,  2, 1, 0.5, ]:
    idx = (mean_log.alpha == alpha) & (mean_log.method == "WNNDAD")
    plt.plot(mean_log.k[idx], mean_log.mae[idx], label = r"$\alpha={}$".format(alpha))

knn_idx = (mean_log.method == "kNN") & (mean_log.alpha == 1)
plt.plot(mean_log.k[knn_idx], mean_log.mae[knn_idx], label = "kNN")

plt.ylim(0.1,1)
plt.hlines(mean_log.mae[mean_log.method == "NNDAD"].mean(), 0, 500, color = "black", linestyle = "--", label = "BRDDE")
plt.legend(fontsize=14)
plt.xlabel("k", fontsize=14)
plt.ylabel("MAE", fontsize=14)
# font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()


plt.savefig("./results/plots/weights.pdf".format(alpha), bbox_inches="tight")
    