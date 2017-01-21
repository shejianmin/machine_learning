import warnings

import numpy as np


def k_means(X, k, max_it):
    num_points, num_dim = X.shape
    data_set = np.zeros((num_points, num_dim + 1))
    data_set[:, :-1] = X

    centroids = data_set[np.random.randint(num_points, size=k), :]
    centroids[:, -1] = range(1, k + 1)

    iterations = 0
    old_centroids = None
    while not should_stop(old_centroids, centroids, iterations, max_it):
        old_centroids = np.copy(centroids)
        iterations += 1
        update_labels(data_set, centroids)
        centroids = get_centroids(data_set, k)

    return data_set


def update_labels(data_set, centroids):
    npm_points, num_dim = data_set.shape
    for i in range(0, npm_points):
        data_set[i, -1] = get_label_from_closest_centroid(data_set[i, :-1], centroids)


def get_label_from_closest_centroid(data_set_row, centroids):
    label = centroids[0, -1]
    min_dist = np.linalg.norm(data_set_row - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(data_set_row - centroids[i, :-1])
        if dist < min_dist:
            min_dist = dist
            label = centroids[i, -1]
    return label


def get_centroids(data_set, k):
    result = np.zeros((k, data_set.shape[1]))
    for i in range(1, k + 1):
        one_cluster = data_set[data_set[:, -1] == i, : -1]
        result[i - 1, :-1] = np.mean(one_cluster, axis=0)
        result[i - 1, -1] = i
    return result


def should_stop(old_centroids, centroids, iterations, max_it):
    if iterations > max_it:
        return True
    return np.array_equal(old_centroids, centroids)


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
test_X = np.vstack((x1, x2, x3, x4))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    result = k_means(test_X, 2, 10)
    print(result)
