import os
import pandas as pd

DATA_PATH = "%s/%s/%s"%(os.path.dirname(os.getcwd()), 'data', 'tidy')

data_set = DATA_PATH + '/tidy_test.csv'
labels = DATA_PATH + '/tidy_label_test.csv'

df = pd.read_csv(data_set, sep=',', header=0, index_col=False)
y = pd.read_csv(labels, sep=',', header=0, index_col=False)

df = pd.concat([df, y],  axis=1)

df_node_2 = df[df['Node_1'] != 1]
df_node_3 = df[df['Node_1'] == 1]

NODE_2_PATH = DATA_PATH + '/node_2/node_2_test.csv'
NODE_3_PATH = DATA_PATH + '/node_3/node_3_test.csv'

df_node_2.to_csv(NODE_2_PATH)
df_node_3.to_csv(NODE_3_PATH)
