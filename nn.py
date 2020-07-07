from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# load the dataset
df_train = np.load('ppdataset_train.npy', allow_pickle='TRUE').item()
df_test = np.load('ppdataset_test.npy', allow_pickle='TRUE').item()

# input size 
input_size = len(df_train['X'].columns)
print('Input size: {}'.format(input_size))

cols_to_del = []
for i in range(0, input_size):
	if (np.isnan(df_train['X'][i]).all()):
		cols_to_del.append(i)

print('Cols to del: ')
print(cols_to_del)	

# output_size
output_size = len(df_train['y'].columns)
print('Output size: {}'.format(output_size))

# num rows 
num_rows = len(df_train['y'])

# size of hidden layer
hidden_layer_size = int((input_size + output_size)/2)

print('Size of hidden layer: {}'.format(hidden_layer_size))

# A neural network with a single hidden layer. 
clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(hidden_layer_size), random_state=1, verbose=True)

nans = 0
rows_to_del = []
# print the dataset 
for i in range(0, num_rows):
	if (np.isnan(df_train['X'].iloc[i,:]).any() or np.isnan(df_train['y'].iloc[i,:]).any()):
		nans += 1
		print('count: {} i: {}'.format(nans, i))
		rows_to_del.append(i)


df_train['X'].drop(df_train['X'].index[rows_to_del], inplace=True)	
df_train['y'].drop(df_train['y'].index[rows_to_del], inplace=True)	
print('Deleted {} rows containing NaN values'.format(nans))		

# recalculate num rows
num_rows = len(df_train['y'])

nans = 0
rows_to_del = []
# print the dataset 
for i in range(0, num_rows):
	if (np.isnan(df_train['X'].iloc[i,:]).any() or np.isnan(df_train['y'].iloc[i,:]).any()):
		nans += 1
		print('count: {} i: {}'.format(nans, i))
		rows_to_del.append(i)

		
#arr = np.array(df_train['y'])
#pr_mean = arr.mean()
#print(pr_mean)
#print(df_train['y'].mean())

df_train['y'] = (df_train['y'] > 0.01).astype(int)
#print(df_train['y'])

#Train the neural network with the training data
clf.fit(df_train['X'], df_train['y'])


