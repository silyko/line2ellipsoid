import sys
import line2ellipsoid as le
import numpy as np
lat,lon = np.radians(map(float, sys.argv[1:3]))
if len(sys.argv) > 3:
    azimuth = np.radians(float(sys.argv[3]))
else:
    azimuth = None
R1 = 6378137.000
R2 = 6356752.314140
#R1 = 1
#R2 = 0.1
n = le.latlon_to_normal(lat, lon)
xyz = le.normal_to_ellipsoid(n, R1, R1, R2)
xyz_geo = xyz / (R1, R1, R2)
glat, glon = np.degrees(le.normal_to_latlon(xyz_geo))
print("Input: lat: %s, lon: %s" % (sys.argv[1], sys.argv[2]))
print("output: x: %.3f, y: %.3f, z: %.3f" % (xyz[0,0], xyz[0,1], xyz[0,2]))
print("Rotational lat,lon parameters (v, u): lat: %.5f, lon: %.5f" % (glat, glon))

    
if azimuth is not None:
    curv = le.sectional_curv(azimuth, R1, R2, lat, lon)
    r_curv = 1 / curv 
    print("Sectional curvature for azimuth: %.2f, curv: %.2f" % (np.degrees(azimuth), curv)) 
    print("Radius of curvature: %.2f" % (r_curv))
