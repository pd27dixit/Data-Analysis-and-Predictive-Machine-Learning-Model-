# Roll Number: 20CS10047
# Project Code: SBHC-DC
# Project Title: Shill Bidding using Complete Linkage Divisive (Top-down) Clustering Technique
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import time


def cosine_similarity(x, y):
    """
    Compute cosine similarity between vectors x and y.
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def k_means_cosine(X, k, max_iter=20):
    """
    Cluster the data X into k clusters using K-means algorithm with cosine similarity as the distance measure.
    """
    n_samples, n_features = X.shape

    # Initialize centroids randomly
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroid_indices]

    # Iterate until convergence or maximum number of iterations reached
    for i in range(max_iter):
        # Assign each point to the nearest centroid
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.apply_along_axis(
                cosine_similarity, 1, X, centroids[j])
        labels = np.argmax(distances, axis=1)

        # Update centroids
        for j in range(k):
            mask = labels == j
            if np.sum(mask) > 0:
                centroids[j] = np.apply_along_axis(np.mean, 0, X[mask])

    return labels


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    # print("intersection ",intersection)
    union = len(set1.union(set2))
    # print("union ",union)
    return intersection / union if union != 0 else 0


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# import heapq
# from scipy.spatial.distance import pdist, squareform
# def complete_linkage(X, k):
#     """
#     Returns the labels obtained from complete linkage clustering.
#     """
#     n = X.shape[0]

#     # Compute pairwise distances between data points
#     distances = squareform(pdist(X))

#     # Initialize clusters
#     clusters = np.arange(n).reshape(-1, 1)

#     # Initialize binary heap with pairwise distances between clusters
#     heap = []
#     for i in range(n):
#         for j in range(i+1, n):
#             heapq.heappush(heap, (distances[i, j], (i, j)))

#     # Merge clusters until k clusters remain
#     while len(clusters) > k:
#         # Pop the two closest clusters from the heap
#         _, (merge_i, merge_j) = heapq.heappop(heap)

#         # Merge the two closest clusters
#         clusters[merge_i] = np.concatenate((clusters[merge_i], clusters[merge_j]))
#         clusters = np.delete(clusters, merge_j, axis=0)

#         # Update distances in the heap
#         for i in range(len(clusters)):
#             if i != merge_i:
#                 dist = np.max(distances[clusters[merge_i], clusters[i]])
#                 heapq.heappush(heap, (dist, (merge_i, i)))

#     # Assign labels based on clusters
#     labels = np.zeros(n)
#     for i, cluster in enumerate(clusters):
#         labels[cluster] = i
#     return labels
"""
import numpy as np
from scipy.spatial.distance import cdist

def complete_linkage(X, k):
    n = X.shape[0]

    # Compute pairwise distances using vectorization
    distances = cdist(X, X)

    # Initialize union-find data structure
    parent = np.arange(n)
    rank = np.zeros(n)

    def find(i, parent):
        # Find the root of the cluster
        root = i
        while root != parent[root]:
            root = parent[root]

        # Compress the path
        while i != root:
            parent[i], i = root, parent[i]

        return root

    # Merge clusters until k clusters remain
    for i in range(n-1):
        # Find the indices of the two closest clusters
        i1, i2 = np.unravel_index(np.argmin(distances), distances.shape)

        # Merge the two closest clusters
        root1 = find(i1, parent)
        root2 = find(i2, parent)
        if root1 != root2:
            union(root1, root2, parent, rank)

        # Update distances
        distances[:, i1] = np.maximum(distances[:, root1], distances[:, root2])
        distances[i1, :] = distances[:, i1]
        distances[:, i2] = np.inf
        distances[i2, :] = np.inf

        if len(np.unique(parent)) == k:
            break

    # Assign labels based on clusters
    labels = np.zeros(n)
    for i in range(n):
        labels[i] = find(i, parent)

    return labels

def find(i, parent):
    if parent[i] != i:
        parent[i] = find(parent[i], parent)
    return parent[i]

def union(i, j, parent, rank):
    if rank[i] < rank[j]:
        parent[i] = j
    elif rank[i] > rank[j]:
        parent[j] = i
    else:
        parent[j] = i
        rank[i] += 1
"""
# import heapq


def complete_linkage(X, k):
    """
    Returns the labels obtained from complete linkage clustering.
    """
    n = X.shape[0]

    # Precompute pairwise distances
    distances = np.zeros((n, n))
    distances = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))

    # Initialize clusters
    clusters = [[i] for i in range(n)]

    # Merge clusters until k clusters remain
    while len(clusters) > k:
        # Find the two closest clusters
        min_dist = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                max_dist = 0
                for ii in clusters[i]:
                    for jj in clusters[j]:
                        max_dist = max(max_dist, distances[ii, jj])
                if max_dist < min_dist:
                    min_dist = max_dist
                    merge_i, merge_j = i, j

        # Merge the two closest clusters
        clusters[merge_i] += clusters[merge_j]
        del clusters[merge_j]

        # Update distances
        for i in range(len(clusters)):
            if i != merge_i:
                max_dist = 0
                for ii in clusters[i]:
                    for jj in clusters[merge_i]:
                        max_dist = max(max_dist, distances[ii, jj])
                distances[i, merge_i] = max_dist
                distances[merge_i, i] = max_dist

    # Assign labels based on clusters
    labels = np.zeros(n)
    for i, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = i

    return labels


def get_cluster_bounds(labels):
    """
    Returns the lower and upper bounds of each cluster in the labels.
    """
    bounds = []
    start = 0
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)[0]
        if len(indices) > 0:
            # subtract 1 from end to get the correct upper bound
            end = start + len(indices) - 1
            bounds.append((start, end))
            start = end + 1  # add 1 to start for the next cluster
    return bounds


start_time = time.time()
data = pd.read_csv('shillbid.csv')
X = data.iloc[:, 3:12].values


# Drop the columns that are not needed
# X = data.drop(['Record_ID', 'Auction_ID', 'Bidder_ID', 'Auction_Duration'], axis=1)


np.random.seed(np.random.seed(63))
kmeans_labels = k_means_cosine(X, k=3, max_iter=20)
print(kmeans_labels)

# print(kmeans_labels)
# print("\n")


s = silhouette_score(X, kmeans_labels, metric='cosine')
print(f"Silhouette coefficient: {s}\n")
max_coeff = s
optimal_k = 3
optimal_k_label = kmeans_labels

# max_coeff = -1
# optimal_k = -1
# optimal_k_label = np.empty(0)


for k in range(4, 7):
    print(f'k = {k}')
    np.random.seed(np.random.seed(63))
    kl = k_means_cosine(X, k=k, max_iter=20)
    # print(kl)
    unique_labels, label_counts = np.unique(kl, return_counts=True)
    # for label, count in zip(unique_labels, label_counts):
    #     print(f'Cluster {label}: {count} instances')
    sc = silhouette_score(X, kl, metric='cosine')
    print("silhouette coefficient ", sc)
    print("\n")
    if (sc > max_coeff):
        max_coeff = sc
        optimal_k = k
        optimal_k_label = kl
print("optimal k ", optimal_k)
print("coefficient ", max_coeff)
print(optimal_k_label)


with open("kmeans.txt", "w") as f:
    for i in range(optimal_k):
        line = ""
        for j in range(len(optimal_k_label)):
            if (optimal_k_label[j] == i):
                line += str(j) + ","
        line = line[:-1]  # remove the last comma
        f.write(line + "\n")

# Complete linkage clustering
agg_labels = complete_linkage(X, optimal_k)
print(agg_labels)
# print("\n")

with open("divisive.txt", "w") as f:
    for i in range(optimal_k):
        line = ""
        for j in range(len(agg_labels)):
            if (agg_labels[j] == i):
                line += str(j) + ","
        line = line[:-1]  # remove the last comma
        f.write(line + "\n")


jaccard_similarities = np.zeros((optimal_k, optimal_k))

# compute sets of data points belonging to each cluster in k-means
kmeans_cluster_sets = []
for i in range(optimal_k):
    cluster_set = set()
    for j in range(len(optimal_k_label)):
        if (optimal_k_label[j] == i):
            cluster_set.add(j)
    # print(cluster_set)
    # print("\n")
    kmeans_cluster_sets.append(cluster_set)

# print(kmeans_cluster_sets)

# compute sets of data points belonging to each cluster in hierarchical clustering

agg_cluster_sets = []
for i in range(optimal_k):
    cluster_set = set()
    for j in range(len(agg_labels)):
        if (agg_labels[j] == i):
            cluster_set.add(j)
    # print(cluster_set)
    # print("\n")
    agg_cluster_sets.append(cluster_set)

print("\n")
# print(agg_cluster_sets)
# compute Jaccard similarity between corresponding sets of k-means and hierarchical clustering

# find one-to-one and onto correspondence of k-means clusters with hierarchical clusters
jaccard_similarities = np.zeros((optimal_k, optimal_k))
for i in range(optimal_k):
    for j in range(optimal_k):
        jaccard_similarities[i, j] = jaccard_similarity(
            kmeans_cluster_sets[i], agg_cluster_sets[j])

# find one-to-one and onto correspondence of k-means clusters with hierarchical clusters
correspondence = [-1] * optimal_k
for i in range(optimal_k):
    max_jaccard_similarity = -1
    max_jaccard_similarity_index = -1
    for j in range(optimal_k):
        if j not in correspondence:
            if jaccard_similarities[i, j] > max_jaccard_similarity:
                max_jaccard_similarity = jaccard_similarities[i, j]
                max_jaccard_similarity_index = j
    correspondence[i] = max_jaccard_similarity_index

# print Jaccard similarity scores and correspondence
print(jaccard_similarities)
for i in range(optimal_k):
    # Find the maximum Jaccard similarity score for this row
    max_score = np.max(jaccard_similarities[i])
    if max_score == 0:
        # Skip this row if all scores are 0
        continue
    # Find the index of the maximum score in this row
    max_index = np.argmax(jaccard_similarities[i])
    if correspondence[i] == max_index:
        # If the maximum score corresponds to the correct cluster, print it in green
        print(f"\033[92mCluster {i} in k-means corresponds to cluster {correspondence[i]} in hierarchical clustering with Jaccard similarity score of {max_score:.3f}\033[0m")
    else:
        # If the maximum score does not correspond to the correct cluster, print it in red
        print(f"\033[92mCluster {i} in k-means corresponds to cluster {correspondence[i]} in hierarchical clustering with Jaccard similarity score of {max_score:.3f}\033[0m")


end_time = time.time()
elapsed_time = end_time - start_time

# Print elapsed time in seconds
print(f'Total time taken: {elapsed_time:.2f} seconds')
