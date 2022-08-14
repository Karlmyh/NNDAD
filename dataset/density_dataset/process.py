import pandas as pd

dataset_seq=["lympho","cardio","thyroid","vowels","glass","musk","letter","pima","satellite","pendigits"]

for dataset in dataset_seq:
    file_path = "../anomaly/{}.csv".format(dataset)
    data = pd.read_csv(file_path,delimiter=",",header=None)
    data=data.drop(columns=data.shape[1]-1)
    data.to_csv("./{}.csv".format(dataset),columns=None)