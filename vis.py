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


#n_clusters = pd.read_pickle('clusters/nclusterdata.pkl')

#plt.plot(n_clusters['time'], n_clusters['n_clusters'])
#plt.title('Number of clusters in each time of day') 
#plt.xlabel('hour of the day')
#plt.ylabel('number of clusters') 
#plt.xticks(rotation=45)

##pickle.dump(ax, open("plot.pickle", "wb"))

#loc = 'clusters/nclusters.png'
#plt.savefig(loc, dpi=1000, bbox_inches='tight')

#loc = 'clusters/nclusters.eps'
#plt.savefig(loc, format='eps', dpi=1000, bbox_inches='tight')

sinr_map = np.load('sinr_map_one_off.npy')

plt.imshow(sinr_map, cmap='viridis')
plt.colorbar()
plt.show()

# Clear the plot
plt.clf()
plt.cla()
plt.close()


