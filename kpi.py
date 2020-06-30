from grid import *
import matplotlib.pyplot as plt 
import geopy.distance
import numpy as np
import shapely.geometry as shape
from scipy.spatial import Voronoi, voronoi_plot_2d
from utils import voronoi_finite_polygons_2d, PolyArea
from mpl_toolkits.mplot3d import Axes3D

class LTESim:
	def __init__(self):
		# setup the class with the default values
		self.point_map = {}
		self.sinr_map = None
		self.density_map = None
		self.grid_conf = {}
		
		self.set_num_flows()
		self.set_think_time()
		self.set_flow_size()
		self.set_fast_fading_margin()
		self.set_antenna_diversity_gain()
		self.set_bs_noise_fig()
		self.set_ue_noise_fig()
		self.set_noise_temp()
		self.set_max_power_tx()
		self.set_topology()
		
		self.set_bw_eff()	
		self.set_sinr_eff()	
		self.set_bw()
		self.set_bs_h()	
		self.set_ue_h()	
		
		self.set_cmax()		
		
		# grid for clove topology
		self.clove_grid = []
		self.y = []
		# coords
		self.coords = None
		self.points = None
		self.inf_map = None
		# get max cell capacity
		self.ci_map = None
		# ro and mew
		self.ro = None
		self.mew = None
		
	# set number of flows that can served parallely
	def set_num_flows(self, val = 2):
		setattr(self, 'num_flows', val)
			
	# set think time of each user in seconds
	def set_think_time(self, val = 2):
		setattr(self, 'think_time', val)	
			
	# set fast fading margin value in dB
	def set_flow_size(self, val = 64):
		setattr(self, 'omega', val)	
		
	# set fast fading margin value in dB
	def set_fast_fading_margin(self, val = -2):
		setattr(self, 'ffg', val)
		
	# set antenna diversity gain in dB	
	def set_antenna_diversity_gain(self, val = 3):
		setattr(self, 'adg', val)	
	
	# set Base Station Noise Figure in dB	
	# Note: Sign has to be taken care of. 
	# For example, if power lost is 5 dB then -5 
	# should be passed as arguments
	def set_bs_noise_fig(self, val=-5):
		setattr(self, 'bs_noise_fig', val)	
	
	# set User Equipment Noise Figure in dB	
	# Note: Sign has to be taken care of. 
	# For example, if power lost is 5 dB then -9
	# should be passed as arguments
	def set_ue_noise_fig(self, val = -9):
		setattr(self, 'ue_noise_fig', val)	
	
	# set the internal noise temperature value
	# Note: This value is in dBm, calculated by N_0*B,
	# where N_0 is the Noise temperature density and B is the bandwidth. 
	# For example, for B = 20 Mhz, value os -100.8174 at 
	# ambient temperature of 300k
	def set_noise_temp(self, val = -100.8174):
		setattr(self, 'N', val)	
	
	# Set maximum power transmit value in dBm
	def set_max_power_tx(self, val = 49):
		setattr(self, 'ptx', val)
		
	# set bandwidth efficiency	'a'
	def set_bw_eff(self, val = 0.63):
		setattr(self, 'a', val)	
	
	# set SINR efficiency	'b'
	def set_sinr_eff(self, val = 0.4):
		setattr(self, 'b', val)	
	
	# set bandwidth in Mhz'B'	
	def set_bw(self, val = 20):
		setattr(self, 'B', val)	
	
	# set max achievable rate in Mbps
	# This is with respect to the wireless 
	# technology/hardware at hand	
	def set_cmax(self, val = 100.8):
		setattr(self, 'cmax', val)
		
	# set BS height in meters	
	def set_bs_h(self, val=32):
		setattr(self, 'bs_h', val)
	
	# set UE height in meters	
	def set_ue_h(self, val=1.5):
		setattr(self, 'ue_h', val)	
		
	# antenna gain, given angle, for each sector.
	# Antenna boresite is the 0 angle, which is the 
	# angle of maximum gain
	# Maximum attentuation is considered to be 20 dB
	def antenna_gain(self, theta, alpha, tilt=0):
		Am = 20
		theta_db = 7*np.pi/18
		alpha_db = 15*np.pi/180
		A_theta = -min(12*np.power(theta/theta_db, np.float(2)), Am)	
		A_alpha = -min(12*np.power((alpha - tilt)/alpha_db, np.float(2)), Am)
		
		combined = -min(-(A_alpha + A_theta), Am)
#		print('Antenna gain {}'.format(combined))
		return combined	
	
	# takes distance in cartesian co-ordinates
	# hex_c tells us the direction of the antenna
	# as it is towards the center of the hexagonal cell
	def get_inf_params(self, p, cell_c, hex_c):
	
		# horizontal
#		m1 = (p.y - cell_c.y)/(p.x - cell_c.x)
#		m2 = (hex_c.y - cell_c.y)/(hex_c.x - cell_c.x)
#		theta = np.arctan(abs((m1-m2)/(1 + m1*m2)))	
		
		a = np.array([p.x, p.y])
		b = np.array([cell_c.x, cell_c.y])
		c = np.array([hex_c.x, hex_c.y])
		
		ba = a - b
		bc = c - b
		
		cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		theta = np.arccos(cos_theta)
		
		# distance
		r = get_distance(p, cell_c)
		# vertical
		if (np.isclose(r, 0)):
			alpha = np.pi/2
		else:
			alpha = np.arctan((self.bs_h - self.ue_h)/r)
		
		return (r, theta, alpha) 		
	
	# path loss function, where d is in meters
	# return in dB
	
	def path_loss(self, d):
		if (d > 0.0000000001):
			return -(128.1 + 37.6*np.log10(d/1000))	
		return -128.1
	
	# calculates link budget, at a distance 
	# r meters from the Base Station (BS)
	def get_prx(self, r, alpha, theta):
		prx = (self.ptx + self.antenna_gain(alpha, theta) + self.path_loss(r)	+ self.ffg 
			+ self.adg + self.bs_noise_fig + self.ue_noise_fig)
		
#		print('prx: {}'.format(prx))
		return prx	
	
	# Convert power in dBm to Pw (power in Watts)
	def dbm_pw(self, p):
		return (np.power(10, float(-3))*np.power(10, p/10))		
	
	# set a network topology centered at (0,0) with a given inter-site
	# distance d in meters and number of rings nrings
	# if nrings = 0, then a single cell topology is created 
	def set_topology(self, d = 500, nrings = 4):
#		r = d/np.sqrt(3)
#		A = np.sqrt(3)*d*d/2
		r = d/3
		A = 3*np.sqrt(3)*r*r/2
		
		setattr(self, 'd', d)
		setattr(self, 'r', r)
		setattr(self, 'A', A)
		setattr(self, 'topo', HexGrid(d, nrings))
	
	# expects a list of objects of type Points, which are used 
	# to decide a uniform value of user density inside the cells.
	# Note that this assumption is just a simplification of the 
	# situation. This function can be changed to represent delta
	# in any manner possible 
	def set_delta(self, coords):
		for t in coords:
			for h in self.topo.hexagons:
				if h.is_inside(t) is True:
					h.num_pts += 1
					break
		
		for h in self.topo.hexagons:
			h.delta = h.num_pts/self.A
	
	# outputs a vector of sinrs` given a vector of etas
	def get_sinr(self, u, eta):
		cells = self.topo.hexagons
		sinr = [None]*len(cells)
		gamma = [None]*len(cells)
		for i in range(0, len(cells)):
			c = cells[i].p
			dist = get_distance(u, c)
			sum = 0
			for j in range(0, len(cells)):
				if j == i:
					continue
				rc = cells[j].p	
				d = get_distance(u, rc)	
				sum += eta[j]*self.get_prx(d)
				
			sum += -self.N 		
			gamma[i] = self.get_prx(dist)/sum
		
		return gamma
		
	
	def get_sinr_nbs(self, u):
		# first find out the cell to which the UE belongs to
		conn_bs = None
		for h in self.topo.hexagons:
			if (h.is_inside(u)):
				conn_bs = h
				break
		
		if (conn_bs is None):
			return -80 # remember to change this
		
		if (self.topo.map[ctok(conn_bs.c)].on is False):
			return -80
			
		dist = get_distance(u, conn_bs.p)
		conn_bs_prx = self.get_prx(dist)
		
#		print('-- Data --')
#		print('Prx connect BS: {}'.format(conn_bs_prx))
		sum = 0
		
		for nbs in conn_bs.get_nbs_keys():
			# check if its neighbor exists
			key = ctok(nbs)
			if (key not in self.topo.map):
				continue
				
			h = self.topo.map[key]
			
			# check if the BS is switched ON
			if (h.on is False):
				continue
				
			p = cube_to_rect(nbs, self.r)
			d = get_distance(u, p)
#			print('Prx nb: {}'.format(self.get_prx(d)))
			sum += self.dbm_pw((self.get_prx(d)))
		
#		print('Sum: {}'.format(sum))
		sinr = self.dbm_pw(conn_bs_prx)/(sum + self.dbm_pw(self.N))	
		sinr = 10*np.log10(sinr)
		
		if (np.isclose(sinr, -3.82211)):
			print('u: ({}, {})'.format(u.x, u.y))
		return sinr
		
	def create_grid(self, lim=2000, steps=50, of_grid_conf='gconf', of_pt_map='grid', of_y='ytick'):
	
		
		self.grid_conf['lim'] = lim
		self.grid_conf['steps'] = steps
		self.grid_conf['du'] = np.power(2*lim/steps, float(2))
		
		np.save(of_grid_conf + '.npy', self.grid_conf)
		
		x = np.linspace(-lim, lim, num=steps)

		y = []

		for i in range(0, len(x)-1):
			y.append((x[i] + x[i+1])/2)
		
		# optimization, traverse the grid and store all points belonging to each cell
		# init list for each hex
		l = self.topo.get_clove_rings()
		
		# to assign a unique identifier to each cell
		hex_id = 0
		self.clove_grid = []
		for cell in l:
			cell_dict = {}
			cell_dict['center'] = cell['center']
			cell_dict['hex'] = []
			
			for i, c in enumerate(cell['hex']):
				hex_dict = {}
				hex_dict['num'] = i
				hex_dict['center'] = c
				hex_dict['obj'] = Hexagon(c, self.d)
				hex_dict['points'] = []
				hex_dict['id'] = hex_id
				hex_id += 1
				cell_dict['hex'].append(hex_dict)
			
			self.clove_grid.append(cell_dict)	
		
		print(hex_id)
		self.sinr_map = np.empty((len(y), len(y)))
		#init with -80 dB 
		self.sinr_map.fill(-60)
		
		# Map each hexagon to its correspodning point
		# Useful while integrating user density with cell capacity
		for i in range(len(y)):
			for j in range(len(y)):
				# iterate through all the cells and sectors
				found = False
				for cell in self.clove_grid:
					for h in cell['hex']:
						if (h['obj'].is_inside(Point(y[i], y[j]))):
							h['points'].append(Point(i,j))
							break
						
					if (found is True):
						break		
		
		np.save(of_pt_map + '.npy', self.clove_grid)
		self.y = y
		np.save(of_y + '.npy', self.y)
		# Construct a SINR map of this topology
#		for i in range(0, len(y)):
#			for j in range(0, len(y)):
#				self.sinr_map[i,j] = self.get_sinr_nbs(Point(y[i], y[j]))
#		
#		np.save(of_sinr_map + '.npy', self.sinr_map)
	
	def get_sinr_clove(self, of_sinr_map='sinr_map', of_inf_map='inf_map'):
		
		self.sinr_map = np.empty((len(self.y), len(self.y)))
		#init with -80 dB 
		self.sinr_map.fill(-60)
		self.inf_map = x = [[{} for i in range(len(self.y))] for j in range(len(self.y))]
		
		k = 1
		for cell in self.clove_grid:
			print('[SINR CLOVE] Processing cell {}'.format(k))
			for h in cell['hex']:
				print('	[SINR CLOVE] Processing hex {}'.format(h['id']))
				for pt in h['points']:
				
					pt_dict = {}
					
					i = pt.x
					j = pt.y
					u = Point(self.y[i], self.y[j])
			
					# calculate prx from current BS
					r, theta, aplha = self.get_inf_params(u, cell['center'], h['center'])
					conn_bs_prx = self.dbm_pw(self.get_prx(r, theta, aplha))
					pt_dict['conn_bs_prx'] = conn_bs_prx
#					print(h['center'].x, h['center'].y, cell['center'].x, cell['center'].y)
					ibs = self.topo.get_inf_bs(cell['center'], h['num'])
					sum = 0
					pt_dict['ibs'] = []
					for bs in ibs:
						ibs_dict = {}
						# for this interfering bs, center of cell and sector
						cell_c = bs['cell_c']
						hex_c = bs['hex_c']
						if (k <= 7):
							hex_id = self.get_hex_id_from_center(cell_c, hex_c, strict=True)
						else:
							hex_id = self.get_hex_id_from_center(cell_c, hex_c, strict=False)
						
						if (hex_id == -1): # do not take into account ibs not present in topology
							continue	
						ibs_dict['hex_id'] = hex_id
						r, theta, aplha = self.get_inf_params(u, cell_c, hex_c)
						# calculate prx for this scenario
						prx = self.dbm_pw(self.get_prx(r, theta, aplha))
						ibs_dict['prx'] = prx
						sum += prx
						pt_dict['ibs'].append(ibs_dict)
						print('		[SINR CLOVE] inf bs id: {} prx: {}'.format(ibs_dict['hex_id'], ibs_dict['prx']))
					sinr = conn_bs_prx/(sum + self.dbm_pw(self.N))	
					sinr = 10*np.log10(sinr)
					self.sinr_map[i,j] = sinr
					self.inf_map[i][j] = pt_dict
			k += 1	
		np.save(of_sinr_map + '.npy', self.sinr_map)
		np.save(of_inf_map + '.npy', self.inf_map)		
	
	def load_grid_conf(self, file='gconf'):
		self.grid_conf = np.load(file+'.npy', allow_pickle='TRUE').item()
#		print(type(self.grid_conf))
	
	def load_pt_map(self, file='grid'):
		self.clove_grid = np.load(file+'.npy', allow_pickle='TRUE')
#		print(type(self.point_map))
		
	def load_sinr_map(self, file='sinr_map'):
		self.sinr_map = np.load(file+'.npy', allow_pickle='TRUE')
#		print(type(self.sinr_map))

	def load_inf_map(self, file='inf_map'):
		self.inf_map = np.load(file+'.npy', allow_pickle='TRUE')
#		print(type(self.inf_map))

	def load_ytick(self, file='ytick'):
		self.y = np.load(file+'.npy', allow_pickle='TRUE')
		
	def load_all(self):
		self.load_grid_conf()
		self.load_pt_map()
		self.load_sinr_map()
		self.load_inf_map()
		self.load_ytick()							
	
	def print_topo(self, cmap='inferno'):
		
		plt.imshow(self.sinr_map, cmap=cmap)
		plt.colorbar()
		plt.show()			
		
		# Clear the plot
		plt.clf()
		plt.cla()
		plt.close()
	
	def set_coords(self, X, x, y, scale_down=20):
		X = X.astype(float)
		X['Latitude'] = X['Latitude'].apply(lambda t: np.sign(t - float(x))*(geopy.distance.vincenty((t, y), (x, y)).m)) 
		X['Longitude'] = X['Longitude'].apply(lambda t: np.sign(t - float(y))*(geopy.distance.vincenty((x, t), (x, y)).m))

		X['Latitude'] = X['Latitude']/scale_down
		X['Longitude'] = X['Longitude']/scale_down
		self.coords = [Point(x,y) for x,y in zip(X['Latitude'], X['Longitude'])]
		self.points = [[x,y] for x,y in zip(X['Latitude'], X['Longitude'])]
		self.points = np.array(self.points)
	
	def print_vor_tes(self, lim=False, traffic_est_scale = 1000000):
	
		vor = Voronoi(self.points)	
		regions, vertices = voronoi_finite_polygons_2d(vor)
		# colorize
		for region in regions:
			polygon = vertices[region]
			x = []
			y = []
			for p in polygon:
				x.append(p[0])
				y.append(p[1])
			A = PolyArea(x,y)
			plt.fill(*zip(*polygon), alpha=0.4)
			tx = sum(x)/len(x)
			ty = sum(y)/len(y)
			plt.text(tx, ty, str(round(traffic_est_scale/A, 2)))

#		plt.plot(self.points[:,0], self.points[:,1], 'ko')
		
		if (lim is True):
			plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
			plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
			
		plt.show()
		# Clear the plot
		plt.clf()
		plt.cla()
		plt.close()
	
	def grid_est_traffic(self, traffic_est_scale = 100000):
		density_list = []
		
		vor = Voronoi(self.points)	
		regions, vertices = voronoi_finite_polygons_2d(vor)
		# colorize
		for region in regions:
			reg_dict = {}
			polygon = vertices[region]
			polyshape = shape.Polygon(polygon)
			x = []
			y = []
			for p in polygon:
				x.append(p[0])
				y.append(p[1])
			A = PolyArea(x,y)
			
			reg_dict['polygon'] = polyshape
			reg_dict['density'] = traffic_est_scale/A
			density_list.append(reg_dict)
		
		
		self.density_map = np.empty((len(self.y), len(self.y)))	
		# iterate through the grid and calculate density for each point
		for i in range(len(self.y)):
			for j in range(len(self.y)):
				point = shape.Point(self.y[i], self.y[j])
				# iterate through the density dict
				for region in density_list:
					if (region['polygon'].contains(point) is True):
						self.density_map[i,j] = region['density']
						break
						
	def get_density(self, i, j):
		return self.density_map[i,j]					
		
	def print_density_map(self):
		fig = plt.figure()
		ax = plt.axes(projection='3d')
#		X = []
#		Y = []
#		Z = []
#		
#		for i in range(len(self.y)):
#			for j in range(len(self.y)):
#				X.append(i)
#				Y.append(j)
#				Z.append(self.density_map[i,j])
#		
#		X = np.array(X)
#		Y = np.array(Y)
#		Z = np.array(Z)	
		X, Y = np.meshgrid(np.arange(len(self.y)), np.arange(len(self.y)))
		Z = []
#		print(X)
#		for x, y in zip(X,Y):
#			Z.append([self.density_map[x[0], x[1]], self.density_map[y[0], y[1]]])
		
		for i in range(len(self.y)):
			l = []
			for j in range(len(self.y)):
				val = self.density_map[X[i][j], Y[i][j]]
				if (val > 10):
					val = 10
				l.append(val)
			
			Z.append(l)
				
		Z = np.array(Z)		
		ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
		ax.set_title('spatial traffic densities')
		
		ax._axis3don = False
		ax.w_zaxis.line.set_lw(0.0)
		ax.set_zticks([])
#		ax.set_zlim(200)
		plt.show()						
		# Clear the plot
		plt.clf()
		plt.cla()
		plt.close()
		
	def get_hex_id_from_center(self, cc, hc, strict=False):
#		print('cc: ({}, {}) hc: ({}, {})'.format(cc.x, cc.y, hc.x, hc.y))
		for cell in self.clove_grid:
			c = cell['center']
#			print('cell compare: ({}, {})'.format(c.x, c.y))
			if not(np.isclose(c.x, cc.x) and np.isclose(c.y, cc.y)):
				continue
			for h in cell['hex']:
				c = h['center']
#				print('hex compare: ({}, {})'.format(c.x, c.y))
				if (np.isclose(c.x, hc.x) and np.isclose(c.y, hc.y)):
					return h['id']
			break		
		
		if (strict is True):
			raise Exception('Invalid hex center in argument. Check your code!')
		else:
			return -1				
	
	def get_eta(self, eta):
		
		if (len(eta) != 57):
			raise Exception('ERROR: eta must be of size 57')
			
		self.ci_map = np.empty((len(self.y), len(self.y))) 
		self.ci_map.fill(0)
			
		k = 1
		hex_id = 0
		
		for cell in self.clove_grid:
			print('>> [ETA SINR] Processing cell {}'.format(k))
			for h in cell['hex']:
				for pt in h['points']:
					i = pt.x
					j = pt.y
					inf = self.inf_map[i][j]
					u = Point(self.y[i], self.y[j])
			
					conn_bs_prx = inf['conn_bs_prx']
					
					sum = 0
					for bs in inf['ibs']:
						# for this interfering bs, center of cell and sector
						id = bs['hex_id']
						prx = bs['prx']
						if (eta[id] < 0):
							print('[DIAG] eta {} id {}'.format(eta[id], id))
#						print('>> hex_id {}'.format(hex_id))
						# calculate prx for this scenario
						sum += eta[id]*prx
					sinr_ratio = conn_bs_prx/(sum + self.dbm_pw(self.N))	
					sinr = 10*np.log10(sinr_ratio)
					self.ci_map[i][j] = (self.a)*(self.B)*min(np.log2(1 + self.b*sinr), self.cmax)
					if (np.isnan(self.ci_map[i][j])):
						print('>> Cell capacity calculation: ({}, {}) sinr_ratio: {} conn_bs_prx: {} sum: {} N: {}'.format(i, j, sinr_ratio, conn_bs_prx, sum, self.dbm_pw(self.N)))
							
			k += 1	
			# we are only processing the inner 21 cells
#			if (k==8):
#				break
		
#		plt.imshow(self.density_map, cmap='inferno')
#		plt.colorbar()
#		plt.show()			
#		
#		# Clear the plot
#		plt.clf()
#		plt.cla()
#		plt.close()
		
		self.mew = [None]*57
		self.ro = [None]*57
		eta = [None]*57
		
		k = 1
		hex_id = 0
		# now integrate and calculate average cell capacity for each of the 57 cells
		for cell in self.clove_grid:
			print('>> [ETA Ci] Processing cell {}'.format(k))
			for h in cell['hex']:
				# calculate sum of traffic densities for each cell
				area_sum = 0
				for pt in h['points']:
					i = pt.x
					j = pt.y
#					u = Point(self.y[i], self.y[j])
					area_sum += self.density_map[i,j]
				
				area_sum = area_sum*self.A
				
				avc = 0	
				lam_av = 1/self.think_time
				lam = 0
				du = 400 # temporary, replace by self.du
				for pt in h['points']:
					i = pt.x
					j = pt.y
#					u = Point(self.y[i], self.y[j])
					res = (self.density_map[i,j]/(area_sum*self.ci_map[i][j]))*du
					if (np.isnan(res)):
						print('>> Found one NaN: density: {} ci: {} point: ({}, {})'.format(self.density_map[i,j], self.ci_map[i][j], i, j))
						lam += self.density_map[i,j]*du
						# [TODO] instead of skipping this, solve the bug
						continue
					avc += 1/res
					lam += self.density_map[i,j]*du
				
				lam = lam*lam_av	
				mew = avc/self.omega
				print('avc = {} lam = {} mew = {}'.format(avc, lam, mew))
				self.mew[h['id']] = mew
				self.ro[h['id']] = lam/mew
				print('ro for hex id {} : {}'.format(h['id'], self.ro[h['id']]))
								
			k += 1	
			# we are only processing the inner 21 cells
#			if (k==8):
#				break
			
		for i in range(57):
			eta[i] = self.ro[i]*np.power(1 - self.ro[i], float(self.num_flows))/(1 - np.power(self.ro[i], float(self.num_flows)))
				
		return eta
				
		
	def solve(self, init_eta):
		
		iter = 1
		tol = 0.01
		prev_eta = init_eta
		new_eta = self.get_eta(prev_eta)
		print('[SOLVE] iteration: {} eta: {}'.format(iter, new_eta))
#		return new_eta  #temporary
		while not(np.isclose(new_eta, prev_eta, atol=tol).all()):
			prev_eta = new_eta
			new_eta = self.get_eta(new_eta)
			iter = iter + 1
			print('[SOLVE] iteration: {} eta: {}'.format(iter, new_eta))
		return new_eta					
#	# Calculate max achievable rate
#	def max_rate(self, u):
#		rate = (self.a)*(self.B)*min(np.log2(1 + self.b*self.get_sinr_nbs(u)),self.cmax)
#	
#	# calculate average rate
#	def avg_rate(self, )	
				
							
