import pandas as pd
file_path = "./yeast.data"
# above .data file is comma delimited
data = pd.read_csv(file_path,delim_whitespace=True,header=None)
data=data.drop(columns=[0,9])
data.to_csv('./yeast.csv')  