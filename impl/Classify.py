import numpy as np
import operator
import math

def zipWith(f, *xss):
    """Combines n lists using an n-ary <f> function that combines the values in each
list at a given position into a singular value. The return type is a list."""
    return [f(*z) for z in zip(*xss)]

def centroid(*xss):
    """Computes the centroid (average for each dimension) of a given list of vectors"""
    if not xss:
        return None
    return zipWith(avg, *xss)

def avg(*xs):
    """Calculates the average of a given list of arguments <*xs>"""
    return np.float32(sum(xs))/len(xs)

def norm_angle_dist(x, y):
    """Computes the angular distance between normalised vectors as (1 - cos(theta))"""
    return 1 - sum(zipWith(operator.mul, x, y))

def cluster(vectors, k, metric = norm_angle_dist, max_iter = 40, attempts = 10, gen_random_centroids = True):
    """Performs k-means clustering <attempts> times, taking the result with the
smallest sum of distances. Clusters given <vectors> into <k> groups, using <metric>
as a distance metric and performs <max_iter> iterations."""
    distsum, labels, centroids = None, None, None

    for i in xrange(attempts):
        d, l, c = kmeans(vectors, k, metric, max_iter, gen_random_centroids)
        if distsum == None or abs(d) < distsum:
            distsum, labels, centroids = abs(d), l, c

    return labels, distsum, centroids

def kmeans(vectors, k, metric, max_iter, gen_random_centroids):
    """Performs k-means clustering of given <vectors> into <k> clusters. Uses
the given <metric> as a distance metric and performs <max_iter> iterations."""

    assert len(vectors) > 0, "The vector needs to contain action frames. This one does not."

    if gen_random_centroids:
        #Create a random centroid for each vector (each row), as (point, label)
        centroids = [(np.random.uniform(-1, 1, vectors[0].size), i) for i in xrange(k)]
    else:
        #Just take the first <k>
        centroids = zip(vectors[:k], xrange(k))

    points = []

    for it in xrange(max_iter):
        points = []
        for vector in vectors:
            dist, label = None, 0
            for point, lbl in centroids:
                if point == None:
                    break
                ndist = metric(point, vector)
                if dist == None or ndist < dist:
                    dist = ndist
                    label = lbl
            points.append((vector, dist, label))
   
        #Calculate new centroids
        groups = [[point[0] for point in points if point[2] == lbl] for lbl in xrange(k)]

        new_centroids = []
        for i in xrange(k):
            new_centroid = centroid(*groups[i]), i

            if not new_centroid[0]:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(new_centroid) 

        #Nothing happened in this step, exit
        if np.array_equal(centroids, new_centroids): break

        centroids = new_centroids
        #centroids = [(centroid(*groups[i]), i) for i in xrange(k)]

    distsum = 0
    labels = []
    #Get the labels and the sum of distances from closest centroids
    for point in points:
        distsum += point[1]
        labels.append(point[2])

    #Extracts the points from the list of (centroid, label)
    centroids = [np.array(cnt[0]) for cnt in centroids]
    return distsum, labels, centroids
