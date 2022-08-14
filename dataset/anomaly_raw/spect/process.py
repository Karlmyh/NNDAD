import pandas as pd
file_path = "./SPECTF.train"
# above .data file is comma delimited
data = pd.read_csv(file_path,delimiter=",",header=None)

file_path = "./SPECTF.test"
data=data.append(pd.read_csv(file_path,delimiter=",",header=None))
data.to_csv('./heart.csv',index=None) 