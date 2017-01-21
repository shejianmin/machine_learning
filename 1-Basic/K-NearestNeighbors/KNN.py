import math


def ComputeEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return d


d_ag = ComputeEuclideanDistance(3, 104, 18, 90)

print d_ag
