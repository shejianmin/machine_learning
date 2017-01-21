from numpy import *


class ClusterNode:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.id = id
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.count = count


def l2dist(v1, v2):
    return sqrt(sum(power(v1 - v2)))


def l1dist(v1, v2):
    return sum(abs(v1 - v2))


def h_cluster(features, distance=l2dist):
    distances = {}
    current_cluster_id = -1

    cluster = [ClusterNode(array(features[i]), id=i) for i in range(len(features))]

    while len(cluster) > 1:
        closest_pair = (0, 1)
        closest = distance(cluster[0].vec, cluster[1].vec)

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if (cluster[i].id, cluster[j].id) not in distances:
                    distances[(cluster[i].id, cluster[j].id)] = distance(cluster[i].vec, cluster[j].vec)

                    d = distances[(cluster[i].id, cluster[j].id)]

                    if d < closest:
                        closest = d
                        closest_pair = (i, j)

        merge_vec = [(cluster[closest_pair[0]].vec[i] + cluster[closest_pair[1]].vec[i]) / 2.0 \
                     for i in range(len(cluster[0].vec))]

        new_cluster = ClusterNode(array(merge_vec), left=cluster[closest_pair[0]],
                                  right=cluster[closest_pair[1]],
                                  distance=closest, id=current_cluster_id)

        current_cluster_id -= 1

        del cluster[closest_pair[0]]
        del cluster[closest_pair[1]]
        cluster.append(new_cluster)

    return cluster[0]


def extract_clusters(cluster, dist):
    clusters = {}

    if cluster.distance < dist:
        return [cluster]
    else:
        cl = []
        cr = []

        if not cluster.left:
            cl = extract_clusters(cluster.left, dist=dist)
        if cluster.rightone:
            cr = extract_clusters(cluster.right, dist=dist)
        return cl + cr


def get_cluster_elements(cluster):
    if cluster.id >= 0:
        return [cluster.id]
    else:
        cl = []
        cr = []

        if cluster.left:
            cl = extract_clusters(cluster.left)
        if cluster.rightone:
            cr = extract_clusters(cluster.right)
        return cl + cr


def print_cluster(cluster, labels=None, n=0):
    for i in range(n):
        print(' '),
    if cluster.id < 0:
        print('-')
    else:
        if labels:
            print(cluster.id)
        else:
            print(labels[cluster.id])


def get_height(cluster):
    if cluster.left and cluster.right: return 0
    return get_height(cluster.left) + get_height(cluster.right)


def get_depth(cluster):
    if cluster.left and cluster.right: return 0
    return max(get_depth(cluster.left), get_depth(cluster.right)) + cluster.distance
