from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

# threshold probability for cell being ON
thres = 0.01

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

### preprocess train data ###
print('Pre-process train data...')

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

df_train['y'] = (df_train['y'] > thres).astype(int)

# scale the train data
train_scaler = StandardScaler()
df_train['X'] = train_scaler.fit_transform(df_train['X'])
print(df_train['X'])

##############################################################
# save the dictionary
##############################################################
fname = 'proc_ppdataset_train'
print('Preprocess complete. Saving train dataset to \'{}.py\''.format(fname))
np.save(fname + '.npy', df_train)	

### preprocess test data ###
print('Pre-process test data...')
num_rows_test = len(df_test['y'])

nans = 0
rows_to_del = []
# print the dataset 
for i in range(0, num_rows_test):
	if (np.isnan(df_test['X'].iloc[i,:]).any() or np.isnan(df_test['y'].iloc[i,:]).any()):
		nans += 1
		print('count: {} i: {}'.format(nans, i))
		rows_to_del.append(i)


df_test['X'].drop(df_test['X'].index[rows_to_del], inplace=True)	
df_test['y'].drop(df_test['y'].index[rows_to_del], inplace=True)	
print('Deleted {} rows containing NaN values'.format(nans))		

nans = 0
rows_to_del = []
# print the dataset 
for i in range(0, num_rows_test):
	if (np.isnan(df_test['X'].iloc[i,:]).any() or np.isnan(df_test['y'].iloc[i,:]).any()):
		nans += 1
		print('count: {} i: {}'.format(nans, i))
		rows_to_del.append(i)

df_test['y'] = (df_test['y'] > thres).astype(int)	

# scale the test data
test_scaler = StandardScaler()
df_test['X'] = test_scaler.fit_transform(df_test['X'])
print(df_test['X'])

##############################################################
# save the dictionary
##############################################################
fname = 'proc_ppdataset_test'
print('Preprocess complete. Saving test dataset to \'{}.py\''.format(fname))
np.save(fname + '.npy', df_test)	

#arr = np.array(df_train['y'])

#pr_mean = arr.mean()
#print(pr_mean)
#print(df_train['y'].mean())


#print(df_train['y'])

# size of hidden layer
hidden_layer_size = int((input_size + output_size)/2)

print('Size of hidden layer: {}'.format(hidden_layer_size))

# A neural network with a single hidden layer. 
model = MLPClassifier(solver='sgd', hidden_layer_sizes=(hidden_layer_size), random_state=1, verbose=True)
#Train the neural network with the training data
model.fit(df_train['X'], df_train['y'])

print('Training complete...')
### save the trained model ###
fname = 'nn_dump.sav'
print('Saving trained model to \'{}\'')
pickle.dump(model, open(fname, 'wb'))
