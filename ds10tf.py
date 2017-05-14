import tensorflow as tf
import pandas as pd
import random
import os
import tempfile
import sys
input_file = sys.argv[1]
# df = pd.read_csv('DS10.csv', usecols = range(1,2709))
# d = df.values
# l= pd.read_csv('DS10.csv', usecols = [0])
# labels = l.values
colcount = 2709#235
noofrecords = 1000#613 #1000
COLUMNS = []
COLUMNS.append('label')
for i in range (0, colcount):
    COLUMNS.append(str(i+1))
filename = 'DS10'

LABEL_COLUMN = "label"
def build_estimator(model_dir, model_type):
    wide_columns = []
    for i in range(1, colcount + 1):
        wide_columns.append(tf.contrib.layers.real_valued_column(str(i)))

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir
                                            , feature_columns=wide_columns
                                            )
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                        feature_columns=wide_columns,
                                        hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=wide_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m
def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in COLUMNS}
     # """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
    
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
#   categorical_cols = {
#       k: tf.SparseTensor(
#           indices=[[i, 0] for i in range(df[k].size)],
#           values=df[k].values,
#           dense_shape=[df[k].size, 1])
#       for k in CATEGORICAL_COLUMNS}
#   # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    #feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df['label'].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def train():
    df_train = pd.read_csv(
      tf.gfile.Open('1.csv'),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
    df_test = pd.read_csv(
      tf.gfile.Open('2.csv'),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      engine="python")
      
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    #print df_train
    # print df_train
    # df_train = pd.read_csv('1.csv', usecols = range(1,2709))
    # df_test = pd.read_csv('1.csv', usecols = range(1,2709))
    # df_trainl = pd.read_csv('1.csv', usecols = [0])
    # df_testl = pd.read_csv('2.csv', usecols = [0])

    model_dir = 'tmp'#tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, 'wide')
    m.fit(input_fn=lambda: input_fn(df_train), steps=200)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

def shuffle():
    with open(filename + ".libsvm", mode="r") as myFile:
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
        if(count<=0.8*noofrecords):
            thefile1.write("%s" % item)
        elif(count<=noofrecords):
            thefile2.write("%s" % item)
        else:
            thefile3.write("%s" % item)
        count += 1
    thefile1.close()
    thefile2.close()
    thefile3.close()
# filename_queue = tf.train.string_input_producer('DS10.csv')
# reader = tf.TextLineReader(skip_header_lines=0)
shuffle()
os.system("python libsvm2csv.py " + filename + ".libsvm 1.csv " + str(colcount))
os.system("python libsvm2csv.py " + input_file + " 2.csv " + str(colcount))
# os.system("python libsvm2csv.py 2.txt 2.csv 2709")
# os.system("python libsvm2csv.py 3.txt 3.csv 2709")
train()
