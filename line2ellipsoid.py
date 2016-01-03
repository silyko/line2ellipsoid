import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
DEBUG = True
colors = ["red", "blue", "green"]


def plot_in_3d(point_sets, line_sets, spoint_sets = ()):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i,points in enumerate(point_sets):
        ax.scatter(points[:,0], points[:,1], points[:,2], color = colors[i % len(colors)])
    for line in line_sets:
        ax.plot(line[:,0], line[:,1], line[:,2])
    for i, point in enumerate(spoint_sets):
        ax.scatter(point[0], point[1], point[2], s=100, color = colors[(i+1) % len(colors)])
    ax.set_aspect('equal')
    plt.show()


def project_onto_ellipse(x, y, w):
    # Just solve a complex quartic
    a = w[0]
    b = w[1]
    z = complex(x,y)
    p = None
    coeffs = np.array([a**2 - b**2, 2 * complex(-a * x, b * y),0.0 ,2 * complex(a * x,b * y), b**2 - a**2])
    roots = np.roots(coeffs)
    dmin = None
    for r in roots:
        if abs(1 - np.absolute(r)) < 1e-4:
            # There should be two relevant roots
            q = complex(a*np.real(r), b*np.imag(r))
            d = np.absolute(z-q)
            if d < dmin or dmin is None:
                p = q
                dmin = d
    return (np.real(p), np.imag(p)), dmin
    

def project_onto_ellipse2(x, y, w):
    """Brute force projection onto an ellipse with axes w[0], w[1] of a point (x,y).
    Substitute with any better, well known, procedure!"""
    print("Projecting x: %.6f, y: %.6f, x/y: %.6f" % (x,y, x/y))
    theta = np.arctan2(y,x)
    delta = np.radians(5.0 * np.max(w) / np.min(w)) # Hmmm...
    iterations = 0
    dot = 1e6
    last_dot = 1e11
    p = None
    dist = None
    while dot <= last_dot and iterations < 20:
        last_dot = dot
        print("Iteration: %d, Curent angle: %.6f, current dot: %.6f" % (iterations, np.degrees(theta), dot))
        print("Delta: %.4f" % np.degrees(delta))
        vs = np.linspace(theta - delta,theta + delta, 100000)
        ex = w[0]*np.cos(vs)
        ey = w[1]*np.sin(vs)
        E = np.column_stack((ex,ey))
        normals = normalise(E / w**2)
        dots = np.fabs((normals*(np.fliplr(E-(x,y))*(1,-1))).sum(axis=1)) # simplify this...
        i = np.argmin(dots)
        delta *= 0.75
        if dots[i] < last_dot:
            theta = vs[i]
            p = E[i]
            dot = dots[i]
            dist = np.sqrt(((p-(x,y))**2).sum())
        print("DISTANCE is now: %.8f, dot: %.8f" % (dist, dot))
        iterations += 1
    return p, dist

def project_onto_complement(n, x):
    """x should be an array of shape (n,3)"""
    # Project a point x onto plane normal to n
    return x - (x*n).sum(axis=1).reshape((x.shape[0],1))*n


def ellipsoidal_xyz_to_normals(xyz, a, b, c):
    xyz = transform_to_ellipsoid(xyz, 1/a**2, 1/b**2, 1/c**2)
    return normalise(xyz)
    

def normalise(x):
    # I hate numpy.dot - but should probably use it :-/
    n = np.sqrt((x**2).sum(axis=1)).reshape((x.shape[0],1))
    return x/n

def transform_to_ellipsoid(n, a, b, c):
    """Map some unit vectors of shape (n,3) to ellipsoidal points.
    Coordinates in standard basis.
    Just a linear, self adjoint map.
    """
    return n*np.array([[a,b,c]])

def basis_to_onb(b1, b2):
    """Compute planar orthonormal basis from two basis vectors"""
    e1 = normalise(b1)
    e2 = normalise(project_onto_complement(e1, b2))
    return e1, e2

def get_some_points_on_ellipsoid(a,b,c):
    # get some ellisoidal points.
    ns = []
    for lon in np.linspace(0, 2*np.pi, 40):
        for lat in np.linspace(-np.pi*0.5, np.pi*0.5, 30):
            r = np.cos(lat)
            z = np.sin(lat)
            x = np.cos(lon)*r
            y = np.sin(lon)*r
            ns.append((x,y,z))
    ns = np.array(ns)
    return transform_to_ellipsoid(ns, a, b, c)

def stuffit(a, b ,c, x1, x2):
    """Return a normal vector for the closest point, as well as distance.
    a,b,c are ellipsoidal axes, x1, x2 endpoints of a line.
    """
    x1 = np.array(x1).reshape((1,3))
    x2 = np.array(x2).reshape((1,3))
    line = np.vstack((x1,x2))
    # Hey, first get the normals
    n = normalise(x2 - x1)
    print "norm is", (n**2).sum(axis=1)
    # We now have a nice normal vector...
    # Parametrise the plane orthogonal to that
    if abs(n[0,2]) > 0.2:
        # Hmmm - improve here...
        e1 = n + np.array([[10, 0, 0]])
    else:
        e1 = n +  np.array([[0, 0, 10]])
    if DEBUG:
        print("Was: %s" % str(n))
        print("Now: %s" % str(e1))
    e1 = normalise(project_onto_complement(n, e1))
    if DEBUG:
        print("in plane: %s" % str(e1))
    e2 = np.cross(n, e1)
    miss= np.fabs(e2 - project_onto_complement(n, e2))
    assert (np.fabs(miss) < 1e-5).all()
    if DEBUG:
        print("miss: %.6f" % miss.max())
        print("Basis:")
        print("e1: %s" % str(e1))
        print("e2: %s" % str(e2))
        print("e1 dot e2: %.7f " % (e1*e2).sum())
        print("norm e2: %.7f" % (e2**2).sum())
        print("norm e1: %.7f" % (e1**2).sum())
        print("norm n: %.7f" % (n**2).sum())
        print("normal: %s" % str(n))
    
    # Now find a basis for the 'in-between' plane, where points on 
    # ellipsoid, and normals, are parametrised by unit vectors
    te1 = transform_to_ellipsoid(e1, a, b, c)
    te2 = transform_to_ellipsoid(e2, a, b, c)
    te1, te2 = basis_to_onb(te1, te2)
    if DEBUG:
        # Get some points on the ellipsoid, if we wanna do some plotting
        ellipse = get_some_points_on_ellipsoid(a, b, c)
        # Parametrise some normals in that plane.
        vs = np.linspace(0, 2*np.pi, 10000).reshape((10000,1))
        ns = np.cos(vs)*te1 + np.sin(vs)*te2
        on_ellipsoid = transform_to_ellipsoid(ns, a, b, c)
        in_plane = project_onto_complement(n, on_ellipsoid)
    if DEBUG:
        print("ONB for TP:")
        print("e1: %s, e2: %s" % (str(te1), str(te2)))
        print("norms: %.6f %.6f, dot: %.6f" % ((te1**2).sum(), (te2**2).sum(), (te1*te2).sum()))
    # The mapping e1 -> te1, e2-> te2 gives an isometric identification of the two planes.
    # OK, find the matrix for the composite map ( to_ellipsoid -> project_onto_complement)
    # Going from T(P) to P.
    # We are using the basis for P determined above {e1, e2}, and {te1, te2} for T(P)
    # Image vectors:
    pe1 = project_onto_complement(n, transform_to_ellipsoid(te1, a ,b, c))
    pe2 = project_onto_complement(n, transform_to_ellipsoid(te2, a, b, c))
    # Coefficients of composite wrt current basis {e1, e2} (and identified as T(P))
    a11 = (pe1*e1).sum()
    a21 = (pe1*e2).sum()
    a12 = (pe2*e1).sum() 
    a22 = (pe2*e2).sum()
    sym_dif = ((pe1*e2).sum() - (pe2*e1).sum())
    print("IS SYMMETRIC: %.6f" % sym_dif)
    # OK - so not symmetric...
    A = np.array(((a11,a12),(a21,a22)))
    u, w, v = np.linalg.svd(A)
    print u, "U"
    print v, "V"
    print("Singular values: %.7f, %.7f" % (w[0],w[1]))
    # EIGENSPACE MIGHT HAVE MULTIPLICITY 2 (a sphere)
    # So: just hat the first eigenvector, instead of using the computed one!
    # These are the eigenvectors expressed in the 'standard' basis
    ee1 = u[0,0]*e1+u[1,0]*e2
    ee2 = u[1,0]*e1-u[0,0]*e2
    # corresponding basis of TP:
    tee1 = v[0,0]*te1+v[1,0]*te2
    tee2 = v[1,0]*te1-v[0,0]*te2
    if DEBUG:
        print("NEW ONB:")
        print("e1: %s" % str(ee1))
        print("ee2: %s" % str(ee2))
        print("DOTS: %.8f %.8f" % ((ee1*n).sum(), (ee2*n).sum()))
        print("NORMS %.8f %.8f" % ( (ee1**2).sum(), (ee2**2).sum()))
        print("ONB? %.8f" % (ee1*ee2).sum())
    # Get the 'line point' in the orthogonal plane...
    L = project_onto_complement(n, x1)
    # Just checking...
    miss= np.fabs((L -project_onto_complement(n, x2)))
    assert miss.max() < 1e-5
    # compute x,y in 'eigenbasis' - same coords up in TP... 
    x = (ee1*L).sum()
    y = (ee2*L).sum()
    #Try to minimize - brute force
    p, dist = project_onto_ellipse(x, y, w)
    if DEBUG:
        # Just debugging
        xp = (in_plane * ee1).sum(axis=1)
        yp = (in_plane * ee2).sum(axis=1)
        plt.plot(xp, yp)
        plt.scatter(x, y, s=20)
        plt.scatter(p[0], p[1], s=49)
        plt.axis("equal")
        plt.show()
    # Coordinates of normal in 'eigenbasis' - using the "inverse" transform
    # Use q inverse
    
    n1 = p[0] / w[0]
    n2 = p[1] / w[1]
    # We now use the identification of P and TP
    unit_vector = n1*tee1 + n2*tee2
    assert abs(np.sqrt((unit_vector**2).sum()) - 1) < 1e-5
    final_point = transform_to_ellipsoid(unit_vector, a, b, c)
    if DEBUG:
        plot_in_3d([ellipse],[line],[final_point[0]])
    final_normal = normalise(transform_to_ellipsoid(unit_vector, 1/a, 1/b, 1/c))
    final_dot = (final_normal*n).sum()
    print("FINAL DOT: %.6f" % final_dot)
    return final_normal, dist
    

def latlon_to_normal(lat, lon):
    r = np.cos(lat)
    z = np.sin(lat)
    x = np.cos(lon)*r
    y = np.sin(lon)*r
    n = np.array((x,y,z)).reshape((1,3))
    assert abs(np.sqrt((n**2).sum()) -1 ) < 1e-5
    return n

def normal_to_latlon(n):
    r =np.sqrt( n[0,0]**2+n[0,1]**2)
    lat = np.arctan2(n[0,2], r)
    lon = np.arctan2(n[0,1], n[0,0])
    return lat, lon
    

def normal_to_ellipsoid(n, a, b, c):
    xyz = transform_to_ellipsoid(n, a**2, b**2, c**2)
    l = np.sqrt(((xyz/(a,b,c))**2).sum())
    return xyz/l


if __name__ == "__main__":
    lat = 10.82
    lon = 55.17
    d= 100.0
    print("Starting with dist: %.3f, lat: %.5f lon: %.5f" % (d, lat, lon))
    a,b,c = 4000.0, 3600.0, 3000.0
    n1 = latlon_to_normal(np.radians(lat), np.radians(lon))
    v= np.cross(n1,(-2,7,3))
    # just checking
    xyz = normal_to_ellipsoid(n1, a, b, c)
    n2 = ellipsoidal_xyz_to_normals(xyz, a, b, c)
    assert np.fabs(n1-n2).max() < 1e-6
    p0 = xyz + d*n1
    p1 = p0 + 1000.0*v
    p2 = p0 - 1000.0*v
    n, dist = stuffit(a, b, c, p1, p2)
    print("p1: %s, p2: %s" % (str(p1), str(p2))) 
    print("Output normal: %s, dot-line: %.6f" % (n, ((p1 - p2)*n).sum()))
    lat, lon = np.degrees(normal_to_latlon(n))
    print("Output lat: %.5f, lon: %.5f, dist: %.5f" % (lat, lon, dist))
    
    
    
    
    
    
