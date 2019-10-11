import numpy as np
import matplotlib as plt

def dist(p1, p2):
    return np.sum((p2 - p1) ** 2) ** 0.5


def knn(x, y, test_point, k=5):
    # plt.scatter(test_point[0], test_point[1])
    h = []

    m = x.shape[0]
    for i in range(m):

        current_dist = dist(test_point, x[i])
        h.append((current_dist, y[i]))
    h.sort()
    h = np.array(h[:k])
    h = h[:, 1]


    uniq, cnts = np.unique(h, return_counts=True)
    idx = np.argmax(cnts)
    pred = uniq[idx]
    return int(pred)