import pandas as pd
import matplotlib.pyplot as plt
import geopy.distance
from scipy.spatial import Voronoi, voronoi_plot_2d

from kpi import *
from grid import *

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPoint, Point

# ---- Voronoi finite plot 2D -----
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
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
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

# ---- end voronoi finite plot 2d ----    
x = 35.681308
y = 139.767127

X = pd.read_pickle('data22.pkl') # an example dataset
X = X.astype(float)
X['Latitude'] = X['Latitude'].apply(lambda t: np.sign(t - float(x))*(geopy.distance.vincenty((t, y), (x, y)).m)) 
X['Longitude'] = X['Longitude'].apply(lambda t: np.sign(t - float(y))*(geopy.distance.vincenty((x, t), (x, y)).m))

X['Latitude'] = X['Latitude']/20
X['Longitude'] = X['Longitude']/20

coords = [Point(x,y) for x,y in zip(X['Latitude'], X['Longitude'])]
points = [[x,y] for x,y in zip(X['Latitude'], X['Longitude'])]

points = np.array(points)
# compute Voronoi tesselation
vor = Voronoi(points)

# get regions and vertices
# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

eta = [0.5]*57

d = 500
grid = HexGrid(d, 2)

l = grid.get_clove_rings()

#bbox = Polygon() # empty geometry
bpts = []
for cell in l:
	# center is cell['center']
	#plt.scatter(cell['center'].x, cell['center'].y, marker='1', color='black', s=200)
	for c in cell['hex']:
		hex = Hexagon(c, d)
		#bx, by = hex.plt_coords()
		#plt.plot(bx, by, color='black')
		x, y = hex.ext_coords()
		for p,q in zip(x,y):
			bpts.append([p,q])
#		bbox = bbox.union(Polygon(pts))
		
#polygon = [p for p in bbox.exterior.coords]	
#allparts = [p.buffer(0) for p in bbox.geometry]
#polygon.geometry = shapely.ops.cascaded_union(allparts)
#x, y = polygon.geometry.exterior.xy  # here happens the error	
#plt.plot(x, y)
#plt.show()		
#print(bpts)
# get the bounding polygon
box = MultiPoint(bpts).convex_hull

#x,y = box.exterior.xy
#plt.plot(x,y)
#plt.show()

# colorize
for region in regions:
    polygon = vertices[region]
    # Clipping polygon
    poly = Polygon(polygon)
    poly = poly.intersection(box)
    polygon = [p for p in poly.exterior.coords]

    plt.fill(*zip(*polygon), alpha=0.4)

plt.plot(points[:, 0], points[:, 1], 'ko')
plt.axis('equal')
plt.show()


