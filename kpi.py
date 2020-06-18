from grid import *
import matplotlib.pyplot as plt 

class LTESim:
	def __init__(self):
		# setup the class with the default values
		self.point_map = {}
		self.sinr_map = None
		self.grid_conf = {}
		
		self.set_fast_fading_margin()
		self.set_antenna_diversity_gain()
		self.set_bs_noise_fig()
		self.set_ue_noise_fig()
		self.set_noise_temp()
		self.set_max_power_tx()
		self.set_topology()
		

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
	
	# path loss function, where d is in meters
	# return in dB
	
	def path_loss(self, d):
		if (d > 0.0000000001):
			return -(128.1 + 37.6*np.log10(d/1000))	
		return -128.1
	
	# calculates link budget, at a distance 
	# r meters from the Base Station (BS)
	def get_prx(self, r):
		return (self.ptx + self.path_loss(r)	+ self.ffg 
			+ self.adg + self.bs_noise_fig + self.ue_noise_fig)
	
	# Convert power in dBm to Pw (power in Watts)
	def dbm_pw(self, p):
		return (np.power(10, float(-3))*np.power(10, p/10))		
	
	# set a network topology centered at (0,0) with a given inter-site
	# distance d in meters and number of rings nrings
	# if nrings = 0, then a single cell topology is created 
	
	def set_topology(self, d = 500, nrings = 4):
		r = d/np.sqrt(3) 
		A = np.sqrt(3)*d*d/2
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
		
	def create_grid(self, lim=2000, steps=50, of_grid_conf='gconf', of_pt_map='grid', of_sinr_map='sinr_map'):
	
		self.grid_conf['lim'] = lim
		self.grid_conf['steps'] = steps
		
		np.save(of_grid_conf + '.npy', self.grid_conf)
		
		x = np.linspace(-lim, lim, num=steps)

		y = []

		for i in range(0, len(x)-1):
			y.append((x[i] + x[i+1])/2)
		
		# optimization, traverse the grid and store all points belonging to each cell
		# init list for each hex
		for h in self.topo.hexagons:
			self.point_map[ctok(h.c)] = []
		
		self.sinr_map = np.empty((len(y), len(y)))
		
		# Map each hexagon to its correspodning point
		# Useful while integrating user density with cell capacity
		for i in range(len(y)):
			for j in range(len(y)):
				for h in self.topo.hexagons:
					if (h.is_inside(Point(y[i], y[j])) is True):
						self.point_map[ctok(h.c)].append(Point(i, j)) 
		
		np.save(of_pt_map + '.npy', self.point_map)
		 
		# Construct a SINR map of this topology
		for i in range(0, len(y)):
			for j in range(0, len(y)):
				self.sinr_map[i,j] = self.get_sinr_nbs(Point(y[i], y[j]))
		
		np.save(of_sinr_map + '.npy', self.sinr_map)	
	
	def load_grid_conf(self, file='gconf'):
		self.grid_conf = np.load(file+'.npy', allow_pickle='TRUE').item()
		print(type(self.grid_conf))
	
	def load_pt_map(self, file='grid'):
		self.point_map = np.load(file+'.npy', allow_pickle='TRUE').item()
		print(type(self.point_map))
		
	def load_sinr_map(self, file='sinr_map'):
		self.sinr_map = np.load(file+'.npy', allow_pickle='TRUE')
		print(type(self.sinr_map))
		
	def load_all(self):
		self.load_grid_conf()
		self.load_pt_map()
		self.load_sinr_map()							
	
	def print_topo(self, cmap='viridis'):
		
		plt.imshow(self.sinr_map, cmap=cmap)
		plt.colorbar()
		plt.show()			
		
		# Clear the plot
		plt.clf()
		plt.cla()
		plt.close()
				
							
