import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
from scipy.spatial import Voronoi, voronoi_plot_2d

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

#sim = LTESim()
#sim.set_topology(nrings=4)
#sim.set_delta(coords)

##sim.create_grid()
#sim.load_all()
#print(sim.point_map['(0,0,0)'][0].x)
#sim.print_topo()

points = np.random.rand(10,2) #random
vor = Voronoi(points)


fig = voronoi_plot_2d(vor)
plt.show()





#------- old test code --------#
#print(sim.topo.hexagons[0].is_inside(Point(0, 500.0)))

#print(sim.topo.map['(0,0,0)'].on)

#sim.topo.map['(0,0,0)'].on = False

#print(sim.topo.map['(0,0,0)'].on)

#lim = 2000
#lim = 2000
#s = 20
#x = np.arange(-lim, lim+1, s, dtype=int)
#y = np.arange(-lim, lim+1, s, dtype=int)

#steps = 50
#x = np.linspace(-lim, lim, num=steps)

#y = []

#for i in range(0, len(x)-1):
#	y.append((x[i] + x[i+1])/2)

#print(y)
#print(len(y))

#point_map = {}

## optimization, traverse the grid and store all points belonging to each cell
## init list for each hex
#for h in sim.topo.hexagons:
#	point_map[ctok(h.c)] = []

#sinr_map = np.empty((len(y), len(y)))
#for i in range(x.size):
#	for j in range(y.size):
#		for h in sim.topo.hexagons:
#			if (h.is_inside(Point(y[i], y[j])) is True):
#				point_map[ctok(h.c)].append(Point(i, j)) 
				 				

#for i in range(0, len(y)):
#	for j in range(0, len(y)):
#		sinr_map[i,j] = sim.get_sinr_nbs(Point(y[i], y[j]), eta)

#np.save('sinr_map', sinr_map)
#plt.imshow(sinr_map, cmap='viridis')
#plt.colorbar()
#plt.show()

#print(sim.get_sinr_nbs(Point(100, 100), eta))
#print(sim.get_sinr_nbs(Point(100, 200), eta))

#map = np.empty((x.size, y.size))
#for i in range(x.size):
#    for j in range(y.size):
#        map[i,j] = 46 + sim.get_prx(get_distance(Point(0,0), Point(x[i], y[j])))

##y = [(46 + sim.path_loss(abs(r))) for r in x]
#plt.imshow(map, cmap='viridis')
#plt.colorbar()
##plt.plot(x,y)
#plt.show()
 

## plot the grids
#for h in sim.topo.hexagons:
##	print(h.delta)
#	print(len(h.get_nbs()))
#	for nb in h.get_nbs():
#		x,y = nb.plt_coords()
#		plt.plot(x, y, color='black', linewidth=0.5)
#plt.show()		

#plt.plot(X['Latitude'], X['Longitude'], 'o', color='black', mfc='none', markersize=5)
#plt.xlim(-2500,2500)
#plt.ylim(-2500,2500)
#plt.show()	

