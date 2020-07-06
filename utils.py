import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, MultiPoint, Point

# for more solutions, look at :
# https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp(axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    
#    print(vor.ridge_points)
#    print(all_ridges)
#    print(vor.point_region)
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        
        # a quick hack, sometimes, p1 might not be present in the list
        # so just skip it. Real reason needs to be found out. It may
        # happen that skipping it may not be the most correct way of doing it.
        if (p1 in all_ridges):
        	ridges = all_ridges[p1]
        else:
        	continue	
        	
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

## make up data points
#np.random.seed(1234)
#points = np.random.rand(15, 2)

## compute Voronoi tesselation
#vor = Voronoi(points)

## plot
#regions, vertices = voronoi_finite_polygons_2d(vor)
#print("--")
#print(regions)
#print("--")
#print(vertices)

## colorize
#for region in regions:
#	polygon = vertices[region]
#	x = []
#	y = []
#	for p in polygon:
#		x.append(p[0])
#		y.append(p[1])
#	A = PolyArea(x,y)
#	plt.fill(*zip(*polygon), alpha=0.4)
#	tx = sum(x)/len(x)
#	ty = sum(y)/len(y)
#	plt.text(tx, ty, str(round(1/A, 2)))

#plt.plot(points[:,0], points[:,1], 'ko')
##plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
##plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

#plt.show()

# Create a grid and calculate density for each point

#-------second solution-----------------
## make up data points
#points = np.random.rand(15,2)

## add 4 distant dummy points
#points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)

## compute Voronoi tesselation
#vor = Voronoi(points)

## plot
#voronoi_plot_2d(vor, show_vertices = False)

##fig = plt.figure()
##ax = fig.gca(projection='3d')

## colorize
#for i,region in enumerate(vor.regions):
##	print(region)
#	if (len(region)==0):
#		continue
#	if not -1 in region:
#		polygon = [vor.vertices[i] for i in region]
#		x = []
#		y = []
#		for p in polygon:
#			x.append(p[0])
#			y.append(p[1])
#		A = PolyArea(x,y)
##		plt.plot(x, y, 1/A)	
##		print(A)
#		plt.fill(*zip(*polygon))
##		print(np.where(vor.point_region == i)[0])
##		if (len(np.where(vor.point_region == i)[0]) == 0):
##			continue
##		p = np.where(vor.point_region == i)[0][0]
##		print(p)
#		tx = sum(x)/len(x)
#		ty = sum(y)/len(y)
#		plt.text(tx, ty, str(round(1/A, 2)))

## fix the range of axes
##plt.xlim([0,1]), plt.ylim([0,1])
#plt.legend()

#plt.show()
##plt.show()
