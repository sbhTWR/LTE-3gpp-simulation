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
		self.set_grid()
		self.set_topology()
		

	# set fast fading margin value in dB
	def set_fast_fading_margin(val = -2):
		setattr(self, 'ffg', val)
		
	# set antenna diversity gain in dB	
	def set_antenna_diversity_gain(val = 3):
		setattr(self, 'adg', val)	
	
	# set Base Station Noise Figure in dB	
	# Note: Sign has to be taken care of. 
	# For example, if power lost is 5 dB then -5 
	# should be passed as arguments
	def set_bs_noise_fig(val=-5):
		setattr(self, 'bs_noise_fig', val)	
	
	# set User Equipment Noise Figure in dB	
	# Note: Sign has to be taken care of. 
	# For example, if power lost is 5 dB then -9
	# should be passed as arguments
	def set_ue_noise_fig(val = -9):
		setattr(self, 'ue_noise_fig', val)	
	
	# set the internal noise temperature value
	# Note: This value is in dBm, calculated by N_0*B,
	# where N_0 is the Noise temperature density and B is the bandwidth. 
	# For example, for B = 20 Mhz, value os -100.8174 at 
	# ambient temperature of 300k
	def set_noise_temp(val =- 100.8174):
		setattr(self, 'N', -val)	
	
	# Set maximum power transmit value in dBm
	def set_max_power_tx(self, val = 49):
		setattr(self, 'ptx', val)
	
	# path loss function, where d is in meters
	# return in dB
	def path_loss(self, d):
		return -(128.1 + 37.6*numpy.log10(d/1000))	
	
	# calculates link budget, at a distance 
	# r meters from the Base Station (BS)
	def get_prx(self, r):
		return (self.ptx + self.path_loss(r)	+ self.ffg 
			+ self.adg + self.bs_noise_fig + self.ue_noise_fig)
	
	# set a network topology centered at (0,0) with a given inter-site
	# distance d in meters and number of rings nrings
	# if nrings = 0, then a single cell topology is created 
	def set_topology(self, d = 500, nrings = 4):
		setattr(self, 'topo', HexGrid(d, nrings))
	
	# expcets a list of objects of type Points, which are used 
	# to decide a uniform value of user density inside the cells.
	# Note that this assumption is just a simplification of the 
	# situation. This function can be changed to represent delta
	# in any manner possible 
	def set_delta(self, coords):
		
		
			
	
		
				
			
				
	
			 		
	
		
				
