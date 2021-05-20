import pandas as pd
from scipy import stats as sts
from helpers import customArgmax

def training(data,classes,features,class_name):
  vals_mean = dict()
  vals_sd = dict()
  for feature in features:
    temp_mean = dict()
    temp_sd = dict()
    for cls in classes:
      row_data = data[data[class_name] == cls][feature]
      temp_mean[cls] = row_data.describe().get("mean")
      temp_sd[cls] = row_data.describe().get("std")
    vals_mean[feature] = temp_mean
    vals_sd[feature] = temp_sd
  vals = dict()
  vals["mean"] = vals_mean
  vals["sd"] = vals_sd
  temp = dict()
  for cls in classes:
    temp[cls] = len(data[data[class_name] == cls].index)/len(data.index)
  vals["prior_prob"] = temp
  return vals

def predict(inp,vals,classes,features):
  probs = dict()
  for cls in classes:
    temp = vals["prior_prob"][cls]
    for feature in features:
      temp = temp * sts.norm(vals["mean"][feature][cls],vals["sd"][feature][cls]).pdf(inp[feature])
    probs[cls] = temp
  return probs

def cal_loss(data,training_vals,classes,features,class_name):
  count = 0
  for i in range(150):
    row = data.iloc[i].to_dict()
    pred_vals = predict(row,training_vals,classes,features)
    pred = customArgmax(pred_vals)
    if pred != row[class_name]:
      count += 1
  return ((count*100)/len(data.index))