from __future__ import print_function
import numpy as np
import random


# calculate Euclidean distance
def eucl_dist(vec1, vec2):
    return np.sqrt(np.sum(np.power(vec2 - vec1, 2)))


# init centroids with random samples
def init_centroids(dataset, k):
    num_samples, dim = dataset.shape
    centroids = np.zeros((k, dim))
    init_indexes = random.sample(range(num_samples), k)
    for i in range(k):
        index = init_indexes[i]
        centroids[i, :] = dataset[index, :]
    return centroids


# k-means cluster
def kmeans(dataset, k):
    num_samples = dataset.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assignment = np.zeros((num_samples, 2))
    cluster_updated = True

    ## step 1: init centroids
    centroids = init_centroids(dataset, k)

    while cluster_updated:
        cluster_updated = False
        # for each sample
        for i in range(num_samples):  #range
            min_dist = np.finfo(np.float32).max
            min_index = 0
            # for each centroid
            # step 2: find the centroid who is closest
            for j in range(k):
                distance = eucl_dist(centroids[j, :], dataset[i, :])
                if distance < min_dist:
                    min_dist, min_index = distance, j
# step 3: update its cluster
            if cluster_assignment[i, 0] != min_index:
                cluster_updated = True
                cluster_assignment[i, :] = min_index, min_dist**2


# step 4: update centroids
        for j in range(k):
            pts_in_cluster = dataset[np.nonzero(cluster_assignment[:, 0] == j)[0]]
            centroids[j, :] = np.mean(pts_in_cluster, axis=0)
    return centroids, cluster_assignment

if __name__ == '__main__':
    points = [
        [1, 2],
        [2, 1],
        [3, 1],
        [5, 4],
        [5, 5],
        [6, 5],
        [10, 8],
        [7, 9],
        [11, 5],
        [14, 9],
        [14, 14],
    ]
    dataset = np.array(points, np.float32)
    centroids, _ = kmeans(dataset, 3)
    print(centroids)