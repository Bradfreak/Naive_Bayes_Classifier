from helpers import customArgmax
from naive_bayes import training, predict, cal_loss
from pre_process import pre_proc

def input_from_user(features):
  inp = dict()
  print("Enter the feature values")
  for feature in features:
    print(feature+" :-")
    inp[feature] = float(input())
  return inp

print("Enter the dataset filename")
filename = input()
print("Enter the delimiter used")
deli = input()
train_data = pre_proc(filename, deli)
print("Training started...")
ans = training(train_data["data"], train_data["classes"], train_data["features"], train_data["class_name"])
print("Training finished...")
loss = cal_loss(train_data["data"],ans,train_data["classes"],train_data["features"], train_data["class_name"])
print("Loss percentage = ",loss)
end = 0
while end != 1:
  inp = input_from_user(train_data["features"])
  output = predict(inp, ans, train_data["classes"], train_data["features"])
  print("The probabilities of classes are as follows")
  print(output)
  print("The predicted class is ",customArgmax(output))
  print("Enter '0' to continue and '1' to exit")
  end = int(input())