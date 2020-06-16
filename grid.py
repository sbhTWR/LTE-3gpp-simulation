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
	# this code doesn't work for radius == 0; can you see why?
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
	def __init__(self, p, d):
		self.p = p
		self.d = d
		
		self.x = []
		self.y = []
		
		self.bx = []
		self.by = []
		
		self.get_hex()
		
	def plt_coords(self):
		return (self.bx, self.by)
		
	def bd_coords(self):
		return (self.x, self.y)			
		
	def get_hex(self):
		ang = np.pi/6
		ang = 0
		
		# radius of the hexagon
		r = d/np.sqrt(3)
		
		for i in range(0,7):
			tx = self.p.x + r*np.cos(ang + np.pi*i/3)
			ty = self.p.y + r*np.sin(ang + np.pi*i/3)
			self.bx.append(tx)
			self.by.append(ty)
		
		for i in range(0,6):
			tx = self.p.x + r*np.cos(ang + np.pi*i/3)
			ty = self.p.y + r*np.sin(ang + np.pi*i/3)
			self.x.append(tx)
			self.y.append(ty)		

class HexGrid():
	def __init__(self, p, d, nrings=1):
		self.p = p;
		self.nrings = nrings
		self.d = d
		self.r = d/np.sqrt(3)
		
		self.rings = []
		self.centers = []
		self.hexagons = []
		self.ax = None
		
		self.get_rings()
	
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
		
		res = cube_spiral(center, self.nrings)
		self.centers = [cube_to_rect(p, self.r) for p in res]
		self.hexagons = [Hexagon(c, self.d) for c in self.centers]
		#print(self.hexagons[0].bx)
		#print(len(self.hexagons))	
			
	
							
					
		
x = 0.0
y = 0.0

d = 500 # meters

grid = HexGrid((0,0), d, 4)

# plot the grids
for h in grid.hexagons:
	x,y = h.plt_coords()
	plt.plot(x, y, color='black')

plt.show()	

	
