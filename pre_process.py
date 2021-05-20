import pandas as pd

def pre_proc(filename, delim):
    train_data = dict()
    train_data["data"] = pd.read_csv(filename, delimiter=delim)
    train_data["classes"] = set(train_data["data"].iloc[:,-1])
    train_data["features"] = list(train_data["data"].columns[:-1])
    train_data["class_name"] = train_data["data"].columns[-1]
    return train_data