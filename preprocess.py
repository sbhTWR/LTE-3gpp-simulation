from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import geopy.distance
import matplotlib.pyplot as plt

from joblib import dump, load
import pickle

from kpi import *
from grid import *

data = pd.read_csv('data/tokyo.csv', sep=',',encoding = 'utf8',lineterminator='\n',low_memory=False, dtype=str)
data = data.dropna()

# Maximum number of tweets are on 2016-04-16 and 2016-04-17, so drop all rows related to that date
# They will act as the test data
#data = data.drop(data[data['Date']=='2016-04-16'].index)
#data = data.drop(data[data['Date']=='2016-04-17'].index)
# Very low number of tweets on 2016-04-12, so drop it
data = data.drop(data[data['Date']=='2016-04-12'].index)
#--------------------------------------------------------------------------------#

# get date time object
#data['time'] = datetime.strptime(data['Date']+' '+data['Hour'], '%y-%m-%d %H:%M')
print(data['Hour'])
data['time'] =	(data['Hour']).apply(lambda x: datetime.strptime(x, '%H:%M'))
data = data.set_index('time')
print('Columns in dataset:')
print(list(data.columns.values))

# printing dataframe rows where date is equal to date we want
max = 0
date_m = None
print('Numbers of tweets in each day:')
for date in data['Date'].unique():
	n = len(data[data['Date']==date])
	print('{} on {}'.format(n, date))
	if n > max:
		max = n
		date_m = date

print('Max number of tweets in a day: {} on {}'.format(max, date_m))		 
#print(data['Date'][data['Date']=='2016-04-12'])

#print('Times in a day: ')
#print(data['Hour'][data['Date']=='2016-04-16'].unique())

# Tokyo station: 35.681308, 139.767127
x = 35.681308
y = 139.767127
origin = (x,y)
print('Tokyo station: x={}, y={}'.format(x,y))

#coords_1 = (52.2296756, 21.0122287)
#coords_2 = (52.406374, 16.9251681)

#print(geopy.distance.vincenty(coords_1, coords_2).m)

# find the maximum distance d in the dataset
d = 0
for index, row in data.iterrows():
	coord = (row['Latitude'], row['Longitude'])
	curr = geopy.distance.vincenty(origin, coord).m
	if curr > d:
		d = curr

print('Maximum distance of a geo-tagged tweet from the origin is: {} meters'.format(d))

#quit()

def calc_min(origin, hdf):
	min = float('inf') 
	for index, row in hdf.iterrows():
		if origin[0]==row['Latitude'] and origin[1]==row['Longitude']:
			continue
		coord = (row['Latitude'], row['Longitude'])	
		dist = geopy.distance.vincenty(origin, coord).m
		if dist < min:
			min = dist
	return min
	
def calc_min_pts(origin, hdf, eps):
	sum = 0
	for index, row in hdf.iterrows():
		if origin[0]==row['Latitude'] and origin[1]==row['Longitude']:
			continue
		coord = (row['Latitude'], row['Longitude'])	
		dist = geopy.distance.vincenty(origin, coord).m
		if dist < eps:
			sum = sum + 1
	return sum	

def distance_in_meters(x, y):
    return geopy.distance.vincenty((x[0], x[1]), (y[0], y[1])).m
    
# sort by dates 
#for date in data['Date'].unique():
#	df[date] = data[data['Date']==date]

data = data.groupby(pd.Grouper(freq='60Min'))

##############################################################
# initilize LTE Simulation
##############################################################
eta = [0.5]*57

sim = LTESim()
sim.set_topology(nrings=4)
#sim.set_delta(coords)
#sim.create_grid(steps=200)
sim.load_all()
#sim.get_sinr_clove()
#sim.print_topo()
##############################################################
# iterate through the data and obtain probabilitie
############################################################## 
for key, item in data:
	print('Stats for hour {}'.format(key.strftime("%H:%M:%S")))
	hdf = data.get_group(key)
	# iterate through the list and print number of tweets for each day and each hour
	for date in hdf['Date'].unique():
		tempdf = hdf[hdf['Date']==date]
		n = len(tempdf)
		print('{}: {}'.format(date, n))
#		print(tempdf)
		tempdf = tempdf[['Latitude','Longitude']]
		# for tempdf, convert to co-ordinates
		sim.set_coords(tempdf, x, y)
		#sim.print_vor_tes(True)
		sim.grid_est_traffic(traffic_est_scale = 1000000)
		#sim.print_density_map()
		pr, cvg = sim.solve(eta)
		print(pr)
			

#loc = 'clusters/nclusters.png'
#plt.savefig(loc, dpi=1000, bbox_inches='tight')

#loc = 'clusters/nclusters.eps'
#plt.savefig(loc, format='eps', dpi=1000, bbox_inches='tight')

## Clear the plot
#plt.clf()
#plt.cla()
#plt.close()

