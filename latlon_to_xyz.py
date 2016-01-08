import sys
import line2ellipsoid as le
import numpy as np
lat,lon = np.radians(map(float, sys.argv[1:3]))
R1 = 6378137.000
R2 = 6356752.314140
n = le.latlon_to_normal(lat, lon)
xyz = le.normal_to_ellipsoid(n, R1, R1, R2)
print("Input: lat: %s, lon: %s" % (sys.argv[1], sys.argv[2]))
print("output: x: %.3f, y: %.3f, z: %.3f" % (xyz[0,0], xyz[0,1], xyz[0,2]))
