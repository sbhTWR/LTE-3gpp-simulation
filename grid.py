import numpy as np
import matplotlib.pyplot as plt

# Utility functions for the cartesian and hexgonal co-ordinates 

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
class Hex:
	def __init__(self, q, r):
		self.q = q
		self.r = r		
		
class Cube:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z	

def get_distance(p1, p2):
#	print('Distance between ({},{}) and ({},{})'.format(p1.x, p1.y, p2.x, p2.y))
	return (np.sqrt(np.square(p1.x - p2.x) + np.square(p1.y - p2.y)))			
		
def cube_to_axial(cube):
	q = cube.x
	r = cube.z
	return Hex(q, r)

def axial_to_cube(hex):
	x = hex.q
	z = hex.r
	y = -x-z
	return Cube(x, y, z)	

def cube_to_rect(cube, r):
		v, u, w = cube.x, cube.y, cube.z
		x = -u / 2 + v - w / 2
		y = (u - w) * np.sqrt(3)/2
		return Point(x*r, y*r)

def rect_to_cube(p, r):
#	cubex = (np.sqrt(3)*p.x/3 + p.y/3)/r
#	cubey = -(np.sqrt(3)*p.x/3 - p.y/3)/r
#	cubez = -(2*p.y/(3*r))
	
	cubex = (p.y/np.sqrt(3) - p.x/3)/r
	cubez= -(p.y/np.sqrt(3) + p.x/3)/r
	cubey = -(cubex + cubez)
#	print('x: {} y: {} z: {}'.format(cubex, cubey, cubez))
#	return Cube(np.rint(cubex), np.rint(cubey), np.rint(cubez))
	return Cube(cubex, cubey, cubez)		

# convert a cube object to key to slot into the HexGrid hexagons
def ctok(c):
	return '(' + str(c.x) + ',' + str(c.y) + ',' + str(c.z) + ')' 	

def ptok(p):
	return '(' + str(p.x) + ',' + str(p.y) + ')' 						
		
# directions in which we can travel in hex co-ordinate system
cube_directions = [
	Cube(+1, -1, 0), Cube(+1, 0, -1), Cube(0, +1, -1), 
	Cube(-1, +1, 0), Cube(-1, 0, +1), Cube(0, -1, +1), 
]

def cube_direction(direction):
	return cube_directions[direction]

def cube_add(cube1, cube2):
	return Cube(cube1.x + cube2.x, cube1.y + cube2.y, cube1.z + cube2.z)	

def cube_scale(a, k):
	return Cube(a.x * k, a.y * k, a.z * k)	

def cube_neighbor(cube, direction):
	return cube_add(cube, cube_direction(direction))
		

# function to traverse a ring of hexagons 
# returns a ring of cube 

def cube_ring(center, radius):
	results = []
	cube = cube_add(center, 
			cube_scale(cube_direction(4), radius))
	for i in range(0,6):
		for j in range(0,radius):
			results.append(cube)
			cube = cube_neighbor(cube, i)
	return results	

# function for spiraling into a grid of hexagons
def cube_spiral(center, radius):
	results = [center]
	for k in range(1, radius+1):
		results = results + cube_ring(center, k)
	return results    
    
# classes to store hexagons 
			
class Hexagon():
	def __init__(self, c, d, clove=True):
	
		self.c = c
		
		if (clove is False):
			self.d = d
			self.r = d/np.sqrt(3)
		
		else:
			self.d = d
			self.r = d/3
		
		if (clove is False):
			self.p = cube_to_rect(c, self.r)
		else:
			self.p = c	
		
		self.x = []
		self.y = []
		
		self.bx = []
		self.by = []
		
		self.on = True
		
		#self.nbs = []
		
		self.get_hex()
		#self.get_nbs()
		
		# Number of users inside the cell 
		self.num_pts = 0
		
		# user density in the cell
		self.delta = 0
		
	def plt_coords(self):
		return (self.bx, self.by)
		
	def ext_coords(self):
		return (self.x, self.y)		
	
	def is_inside(self, t):
		l = len(self.bx)
#		print(l)
		for i in range(0,l-1):
			x0 = self.bx[i]
			y0 = self.by[i]	
			
			x1 = self.bx[i+1]
			y1 = self.by[i+1]
			
			m = (y1 - y0)/(x1 - x0)
			# test if center point and test point have the same sign
			val1 = np.sign((self.p.y - y0) - m*(self.p.x - x0))
			check = (t.y - y0) - m*(t.x - x0)
			val2 = np.sign(check)
			if (np.isclose(check,0)):
#				print('first-> Hex center: ({}, {}) Point: ({}, {}) val = {} m = {} x0 = {} y0 = {} m*2 = {} (x0, y0) = ({}, {}), (x1, y1) = ({}, {})'.format(self.p.x, self.p.y, t.x, t.y, check, m, x0, y0, m*2, x0, y0, x1, y1))
				continue
			else:
				if (val1 != val2):
					return False
		
		# It passed all the loops, so should be inside 
#		print('second')
		return True		
		
	def get_hex(self):
		ang = np.pi/6
#		ang = 0
		
		# radius of the hexagon
		#r = self.d/np.sqrt(3)
		
		for i in range(0,7):
			tx = self.p.x + self.r*np.cos(ang + np.pi*i/3)
			ty = self.p.y + self.r*np.sin(ang + np.pi*i/3)
			self.bx.append(tx)
			self.by.append(ty)
		
		for i in range(0,6):
			tx = self.p.x + self.r*np.cos(ang + np.pi*i/3)
			ty = self.p.y + self.r*np.sin(ang + np.pi*i/3)
			self.x.append(tx)
			self.y.append(ty)
			
	def get_nbs(self):
		res = []
		for k in range(0,6):
			res.append(Hexagon(cube_neighbor(self.c, k), self.d))	
		return res
			
	def get_nbs_keys(self):
		res = []
		for k in range(0,6):
			res.append(cube_neighbor(self.c, k))	
		return res				

class HexGrid():
	def __init__(self, d, nrings=1, clove=True):
#		self.p = p;
		self.nrings = nrings
		
		if (clove is False):
			self.d = d
			self.r = d/np.sqrt(3)
		
		else:
			self.d = d
			self.r = d/3
			
		self.rings = []
		self.centers = []
		self.hexagons = []
		
		self.map = {}
		
		self.ax = None
		
		#self.get_rings()
	
#	def deprecated_get_rings(self):
#		# get center cell
#		self.rings.append(Hexagon(self.p, self.d))
#		
#		for i in range(0, 6):
#			tx = self.p[0] + self.d*np.cos(np.pi*i/3)
#			ty = self.p[1] + self.d*np.sin(np.pi*i/3)
#			self.rings.append(Hexagon((tx, ty), self.d))
	
	def get_rings(self):
		center = Cube(0,0,0)
		
		# create a single cell if nrings = 0
		if self.nrings == 0:
			self.hexagons = [Hexagon(center, self.d)]
			self.map[ctok(center)] = Hexagon(center, self.d)
			return
			
		res = cube_spiral(center, self.nrings)
#		self.centers = [cube_to_rect(p, self.r) for p in res]
		self.hexagons = [Hexagon(c, self.d) for c in res]
		for c in res:
			self.map[ctok(c)] = Hexagon(c, self.d)
		#print(self.hexagons[0].bx)
		#print(len(self.hexagons))	
	
	
	
		
	# receive an object of type Point
	def get_clove_cells(self, p):
		rect_c1 = Point(p.x + self.r*np.cos(np.pi/6), p.y + self.r*np.sin(np.pi/6))
#		c1 = rect_to_cube(rect_c1, self.r)
#		c2 = cube_add(c1, cube_direction(3))
#		c3 = cube_add(c1, cube_direction(4))
		d = np.sqrt(3)*self.r
		rect_c2 = Point(rect_c1.x - d, rect_c1.y)
		rect_c3 = Point(p.x, p.y - self.r)
#		print('----')
#		print('x: {} y: {} z: {}'.format(c1.x, c1.y, c1.z))
#		print('x: {} y: {} z: {}'.format(c2.x, c2.y, c2.z))
#		print('x: {} y: {} z: {}'.format(c3.x, c3.y, c3.z))
		return {'center': p  , 'hex': [rect_c1, rect_c2, rect_c3]}
		
		
	# get information about the interfering cells
	# num can be either 1, 2 or 3
	# cell center is returned as POint object,
	# Hexagon centers are returned in cube co-ordinates
	
	def get_inf_bs(self, c, num):
		if (num == 1):
			list_bs = [3,4,5]
		elif (num == 2):
			list_bs = [5,0,1]
		else:
			list_bs = [1,2,3]	
				
		bs = []
		init_angle = -np.pi/6
		for n in list_bs:
			cell_c = Point(c.x + self.d*np.cos(init_angle + n*np.pi/3), 
						   c.y + self.d*np.sin(init_angle + n*np.pi/3))
													
			# calculate center of cell, and the interfering sector 
			hex_c = self.get_clove_cells(cell_c)['hex'][num-1]	
#			print(hex_c.x,hex_c.y,hex_c.z)
			bs.append({'cell_c': cell_c, 'hex_c': hex_c})									
		
		return bs	
		
	# Gets clove rings as in 3GPP LTE FDD topology	
	def get_clove_rings(self):
		center = Point(0,0)
		ring = []
		
		ring.append(self.get_clove_cells(center))
		
		if self.nrings == 0:
			return ring
		
		dist = self.d
		ang_diff = np.pi/(3)
		
		for i in range(0,6):
			init_ang = -np.pi/6
#			init_ang = 0
			ring.append(self.get_clove_cells(Point(dist*np.cos(init_ang + i*ang_diff), dist*np.sin(init_ang + i*ang_diff))))
#			ring.append(self.get_clove_cells(Point(2*dist*np.cos(init_ang + i*ang_diff), 2*dist*np.sin(init_ang + i*ang_diff))))
		
		# a separate loop, just for the sake of maintaining order
		for i in range(0,6):
			init_ang = -np.pi/6
#			init_ang = 0
			ring.append(self.get_clove_cells(Point(2*dist*np.cos(init_ang + i*ang_diff), 2*dist*np.sin(init_ang + i*ang_diff))))	
			
		for i in range(0,6):
#				init_ang = -np.pi/6
			init_ang = 0
			m = np.sqrt(3)
			ring.append(self.get_clove_cells(Point(m*dist*np.cos(init_ang + i*ang_diff), m*dist*np.sin(init_ang + i*ang_diff))))
		
		return ring		
				
		
			
			
				
