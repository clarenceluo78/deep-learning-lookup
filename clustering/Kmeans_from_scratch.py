import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import random

from .metrics import rand_index, silhouette_coef, sensitive_analysis

class KMeans():
    """K-Means clustering.
    """
    def __init__(self, num_clusters=3, max_iterations=100, random_seed=10, show_epochs=True):
        self.K = num_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.show_epochs = show_epochs
        self.iteration = None
        self.centroids = None
        self.num_samples = None
        self.num_features = None
    
    def init_random_centroids(self, X):
        """Initialize random centroids

        Args:
            X (_type_): input data

        Returns:
            _type_: random initialized centroids for each k
        """
        centroids = np.zeros((self.K, self.num_features))
        np.random.seed(self.random_seed)

        for k in range(self.K):
            init_idx = np.random.choice(range(self.num_samples))
            centroids[k] = X[init_idx]
            # print(init_idx)
        if (self.show_epochs):
            print("Choose initial centroids as:\n", centroids)
        
        return centroids
    
    def assign_clusters(self, X, centroids):
        """Create clusters, consist of samples' indices that
        belongs to this cluster.

        Args:
            X (_type_): input data
            centroids (_type_): current centroid

        Returns:
            _type_: clusters with sample's indices
        """
        clusters = [[] for _ in range(self.K)]  # create empty cluster for each k

        for sample_idx, sample in enumerate(X):
            # select centroid index with smallest Euclidean distance
            closest_centroid_idx = np.argmin(np.linalg.norm(sample - centroids, axis=1))
            # closest_centroid_idx = np.argmin(np.sqrt(np.sum( (sample - centroids)**2, axis=1 )))
            # closest_centroid = np.argmin(np.dot(sample - centroids, sample - centroids))
            
            # append sample index to this cluster
            clusters[closest_centroid_idx].append(sample_idx)
            
        return clusters
    
    def calculate_new_centroids(self, X, clusters):
        """Recalculate new centroids with the current clusters

        Args:
            X (_type_): input data
            clusters (_type_): current clusters

        Returns:
            _type_: new centroids
        """
        new_centroids = np.zeros((self.K, self.num_features))
        
        for idx, cluster_idx in enumerate(clusters):
            new_centroid = np.mean(X[cluster_idx], axis=0)  # vertical
            new_centroids[idx] = new_centroid
            
        return new_centroids
    
    def get_cluster_label(self, X, clusters):
        """Get cluster label with the current clusters
        for each sample

        Args:
            X (_type_): input data
            clusters (_type_): current clusters

        Returns:
            _type_: label prediction for each sample datapoint
        """
        y_pred = np.zeros(self.num_samples)
        
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx + 1  # fit to the original cluster value
                
        return np.int64(y_pred)
    
    def fit(self, X):
        """Fit and train the given input X

        Args:
            X (_type_): input data

        Returns:
            _type_: label prediction for each sample datapoint
        """
        self.num_samples, self.num_features = X.shape
        centroids = self.init_random_centroids(X)
        iteration_cnt = 0
        old_clusters = [[] for _ in range(self.K)]
        if (self.show_epochs):
            print("Start training:")
        for i in range(self.max_iterations):
            clusters = self.assign_clusters(X, centroids)  # calculate new clusters
            if (self.show_epochs):
                print("#cluster 1: {}, #cluster 2: {}, #cluster 3: {}".\
                    format(len(clusters[0]), len(clusters[1]), len(clusters[2])))
            
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(X, clusters)
            
            iteration_cnt += 1
            
            # check convergence
            if old_clusters == clusters:
                if (self.show_epochs):
                    print("K means has converged!")
                break
            
            old_clusters = clusters  # store the previous clusters
            
        y_pred = self.get_cluster_label(X, clusters)  # generate cluster label
        self.iteration_cnt = iteration_cnt
        self.centroids = centroids
            
        return np.int64(y_pred)

if __name__ == '__main__':
    seeds = np.loadtxt("seeds_dataset.txt")
    X = seeds[:, :7]
    y = np.int64(seeds[:, 7])

    ''' K-means:
        Runing K-means with multiple random initializations,
        and picking one with the highest
    '''
    print('-------------------K-means-------------------')
    # for seed_ in range(10, 101, 10):
    #     np.random.seed(seed_)
    #     KMeans_ = KMeans(num_clusters=3, max_iterations=100)
    #     y_pred = KMeans_.fit(X)
    tic = time.process_time()
    KMeans_ = KMeans(num_clusters=3, max_iterations=100, random_seed=10)
    y_pred = KMeans_.fit(X)
    toc = time.process_time()

    correct_cnt = (y_pred == y).sum()
    print('----Accuracy = {:.2f}%.'.format(correct_cnt / X.shape[0] * 100))
    print('----Rand index = ', rand_index(y, y_pred))
    print('----Silhouette coefficient = ', silhouette_coef(X, y_pred))
    print('----Number of iterations = ', KMeans_.iteration_cnt)
    print ("----Computation time = " + str(1000 * (toc - tic)) + "ms")
    print('----Predicted class labels:\n', y_pred)
    print('-----------------------------------------------')
    
    print('-------------------Sensitivity Analysis-------------------')
    sensitive_analysis(X, y, repeat_times=5, model='Kmeans')
    print('----------------------------------------------------')