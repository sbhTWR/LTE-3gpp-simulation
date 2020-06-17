from grid import *

class LTESim:
	def __init__(self):
		# setup the class with the default values
		
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
	
	def set_noise_temp(self, val =- 100.8174):
		setattr(self, 'N', -val)	
	
	# Set maximum power transmit value in dBm
	def set_max_power_tx(self, val = 49):
		setattr(self, 'ptx', val)
	
	# path loss function, where d is in meters
	# return in dB
	
	def path_loss(self, d):
		return -(128.1 + 37.6*np.log10(d/1000))	
	
	# calculates link budget, at a distance 
	# r meters from the Base Station (BS)
	def get_prx(self, r):
		return (self.ptx + self.path_loss(r)	+ self.ffg 
			+ self.adg + self.bs_noise_fig + self.ue_noise_fig)
	
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
		
	
	def get_sinr_nbs(self, u, eta):
		# first find out the cell to which the UE belongs to
		conn_bs = None
		for h in self.topo.hexagons:
			if (h.is_inside(u)):
				conn_bs = h
				break
		
		if (conn_bs is None):
			return 0
		
		dist = get_distance(u, conn_bs.p)
		conn_bs_prx = self.get_prx(dist)
		
		sum = 0
		
		for nbs in conn_bs.get_nbs():
			d = get_distance(u, nbs.p)
			sum += self.get_prx(d)
		
		sinr = conn_bs_prx/(sum - self.N)	
		
		return sinr
						
