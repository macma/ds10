from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import os
import random
import theano.tensor as T
import mlp
# mem = Memory("./mycache")


def shuffle():
  with open("DS10.libsvm", mode="r", encoding="utf-8") as myFile:
      lines = list(myFile)
  random.shuffle(lines)
  count = 0
  try:
      os.remove('1.txt')
  except OSError:
      pass
  try:
      os.remove('2.txt')
  except OSError:
      pass
  try:
      os.remove('3.txt')
  except OSError:
      pass
  thefile1 = open('1.txt', 'w')
  thefile2 = open('2.txt', 'w')
  thefile3 = open('3.txt', 'w')
  for item in lines:
    if(count<700):
      thefile1.write("%s\n" % item)
    elif(count<900):
      thefile2.write("%s\n" % item)
    else:
      thefile3.write("%s\n" % item)
    count += 1
  thefile1.close()
  thefile2.close()
  thefile3.close()
shuffle()

def get_data(fileName):
    data = load_svmlight_file(fileName)
    print(type(data))
    return data[0], data[1]
xtrain, ytrain = get_data("1.txt")
xtest, ytest = get_data("2.txt")

# print (a[0])
x = T.dmatrix("x")
print (type(x))