import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
from kpi import *
from grid import *

x = 35.681308
y = 139.767127
X = pd.read_pickle('data22.pkl') # an example dataset
X = X.astype(float)
X['Latitude'] = X['Latitude'].apply(lambda t: np.sign(t - float(x))*(geopy.distance.vincenty((t, y), (x, y)).m)) 
X['Longitude'] = X['Longitude'].apply(lambda t: np.sign(t - float(y))*(geopy.distance.vincenty((x, t), (x, y)).m))

X['Latitude'] = X['Latitude']/20
X['Longitude'] = X['Longitude']/20

coords = [Point(x,y) for x,y in zip(X['Latitude'], X['Longitude'])]
eta = [0.5]*61

sim = LTESim()
sim.set_delta(coords)

#print(sim.topo.hexagons[0].is_inside(Point(475, 60)))

s = 20
x = np.arange(-2000, 2001, s, dtype=int)
y = np.arange(-2000, 2001, s, dtype=int)

sinr_map = np.empty((x.size, y.size))
for i in range(x.size):
    for j in range(y.size):
        sinr_map[i,j] = sim.get_sinr_nbs(Point(x[i], y[j]), eta)

np.save('sinr_map', sinr_map)
plt.imshow(sinr_map, cmap='viridis')
plt.colorbar()
plt.show()

 

## plot the grids
#for h in sim.topo.hexagons:
##	print(h.delta)
#	x,y = h.plt_coords()
#	plt.plot(x, y, color='black', linewidth=0.5)

#plt.plot(X['Latitude'], X['Longitude'], 'o', color='black', mfc='none', markersize=5)
#plt.xlim(-2500,2500)
#plt.ylim(-2500,2500)
#plt.show()	

