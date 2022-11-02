import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import random

from .metrics import rand_index, silhouette_coef, sensitive_analysis

class accKMeans():
    """Accelerated num_clusters-means by triangle inequality
    """
    def __init__(self, num_clusters=3, max_iterations=100, random_seed=10, show_epochs=True):

        self.K = num_clusters
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.show_epochs = show_epochs
        self.X = None
        self.centroids = None
        self.num_samples = None
        self.num_features = None
        self.iteration_cnt = 0
        self.assignments = [0 for i in range(0, len(X))]
        self.lower_bound = None
        self.upper_bound = None
        self.rx = [0 for i in range(0, len(X))]         # the true/false indicator
        
    def init_random_centroids(self):
        """random initialize centroids and upper/lower bounds
        """
        np.random.seed(self.random_seed)

        # pick the initial centroids
        centroids = np.zeros((self.K, self.num_features))
        for k in range(self.K):
            init_idx = np.random.choice(range(self.num_samples))
            centroids[k] = self.X[init_idx]
        self.centroids = centroids
        if (self.show_epochs):
            print("Choose initial centroids as:\n", self.centroids)

        # set each entry of lower bound to be 0
        self.lower_bound = [[0 for i in range(0, self.K)] for j in range(0, self.num_samples)]
        self.upper_bound = [0 for i in range(self.num_samples)]
        
        # assign x to the closest init point
        for i in range(0, len(self.X)):
            closest_distance = float('inf')
            closest_cen = -1
            farest_distance = 0
            sample = self.X[i]
            for centroid_i in range(0, self.K):
                centroid = self.centroids[centroid_i]

                calculate_distance = self.calculate_distance(sample, centroid)
                if (calculate_distance < closest_distance):
                    closest_distance = calculate_distance
                    closest_cen = centroid_i
                if (calculate_distance > farest_distance):
                    farest_distance = calculate_distance

                # each time d(x,c) is computed, set l(x, c) = d(x, c)
                self.lower_bound[i][centroid_i] = calculate_distance
                
            self.assignments[i] = closest_cen

            # assign upper bound u(x) = min_c d(x, c)
            self.upper_bound[i] = closest_distance

    def calculate_distance(self, a, b):
        """calculate distance between two data points

        Args:
            p1 (_type_): _description_
            p2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        length_square = [(a[i] - b[i])**2 for i in range(0, len(a))]
        distance = pow(sum(length_square), 0.5)
        # distance = np.linalg.norm(np.array([p1 - p2]))

        return distance

    def calculate_new_centroids_by_lemmas(self):
        """calculating new centroids by 2 lemmas
        """
        # For each center c, let m(c) be the mean of the points assigned to c
        mc = [0 for j in range(0, self.K)]
        for c in range(0, self.K):
            sum = [0 for i in range(0, self.num_features)]
            amount = 0
            for i in range(0, len(self.X)):
                if (c == self.assignments[i]):
                    sum = [sum[n] + self.X[i][n] for n in range(0, self.num_features)]
                    amount += 1
            if (amount != 0):
                mc[c] = [sum[n]/amount for n in range(0, self.num_features)]
            else:
                mc[c] = self.centroids[c][:]

        # For each point x and center c, assgin l(x, c) = max(l(x, c) - d(c, m(c)), 0)
        for i in range(0, len(self.X)):
            for c in range(0, self.K):
                self.lower_bound[i][c] = max([self.lower_bound[i][c]-self.calculate_distance(self.centroids[c], mc[c]), 0])

        # For each point x, assign
        for i in range(0, len(self.X)):
            self.upper_bound[i] = self.upper_bound[i] + self.calculate_distance(mc[self.assignments[i]], self.centroids[self.assignments[i]])
            self.rx[i] = 1
        
        # Replace each center c by m(c)
        for c in range(0, self.K):
            self.centroids[c] = mc[c][:]

        # for all centers c and c', compute d(c, c')
        dcc = [[0 for j in range(0, self.K)] for i in range(0, self.K)]
        for i in range(0, self.K):
            for j in range(0, self.K):
                if (i == j): 
                    dcc[i][j] = float('inf')     # set upper bound
                else:
                    dcc[i][j] = self.calculate_distance(self.centroids[i], self.centroids[j])

        
        # compute s(c) = 1/2min_{c!=c'}d(c, c')
        sc = [0 for i in range(0, self.K)]
        for i in range(0, self.K):
            sc[i] = min(dcc[i]) / 2

        for i in range(0, len(self.X)):

            # Identify all points x such that u(x) <= s(c(x))
            cx = self.assignments[i]
            if (self.upper_bound[i] <= sc[cx]):     # Lemma 1
                pass

            # For all remaining points x and centers c:
            else:
                for c in [0, 1, 2]:
                    # if three conditions satisfy
                    if (c != cx and self.upper_bound[i] > self.lower_bound[i][c] and self.upper_bound[i] > self.calculate_distance(self.centroids[cx], self.centroids[c])/2):
                        if (self.rx[i] == 1):   # 3a. if r(x) then compute d(x, c(x)) and assign r(x) = false
                            d_x_cx = self.calculate_distance(self.X[i], self.centroids[cx])
                            self.rx[i] = 0
                            # update upper bound
                            self.upper_bound[i] = d_x_cx
                        else:              # 3a. otherwise, d(x, c(x)) = u(x)
                            d_x_cx = self.upper_bound[i]
                        
                        if (d_x_cx > self.lower_bound[i][c] or d_x_cx > 1/2 * self.calculate_distance(self.centroids[cx], self.centroids[c])):
                            d_x_c = self.calculate_distance(self.X[i], self.centroids[c])
                            self.lower_bound[i][c] = d_x_c

                            if (d_x_c < d_x_cx):
                                self.assignments[i] = c
                                cx = c

    def get_cluster_cnt(self):
        """get the number of samples in each cluster

        Returns:
            _type_: _description_
        """
        assignment = self.assignments
        cnt_0, cnt_1, cnt_2 = 0, 0, 0
        for i in range(0, len(assignment)):
            if (assignment[i] == 0):
                cnt_0 += 1
            elif (assignment[i] == 1):
                cnt_1 += 1
            elif (assignment[i] == 2):
                cnt_2 += 1
        return [cnt_0, cnt_1, cnt_2]

    def fit(self, X):
        """train and fit accelerated model

        Args:
            X (_type_): input
        """
        self.X = X
        self.num_samples, self.num_features = X.shape
        
        # initialize the assignments
        self.init_random_centroids()
        old_assignments = self.assignments[:]
        current_assignment = self.get_cluster_cnt()
        
        # start training
        if (self.show_epochs):
            print("Start training:")
            print("#cluster 1: {}, #cluster 2: {}, #cluster 3: {}".format(current_assignment[0], current_assignment[1], current_assignment[2]))
        self.calculate_new_centroids_by_lemmas()
        new_assignments = self.assignments[:]
        current_assignment = self.get_cluster_cnt()
        if (self.show_epochs):
            print("#cluster 1: {}, #cluster 2: {}, #cluster 3: {}".format(current_assignment[0], current_assignment[1], current_assignment[2]))
        
        self.iteration_cnt += 1
        
        # check the first two epochs, repeat till convergence
        while (old_assignments != new_assignments):
            old_assignments = new_assignments[:]
            self.calculate_new_centroids_by_lemmas()
            new_assignments = self.assignments[:]
            current_assignment = self.get_cluster_cnt()
            if (self.show_epochs):
                print("#cluster 1: {}, #cluster 2: {}, #cluster 3: {}".format(current_assignment[0], current_assignment[1], current_assignment[2]))
            
            self.iteration_cnt += 1
        
        if (self.show_epochs):
            print("Accelerated K-means has converged!")

if __name__ == '__main__':
    seeds = np.loadtxt("seeds_dataset.txt")
    X = seeds[:, :7]
    y = np.int64(seeds[:, 7])
    
    ''' Accelerated K-means:
        Runing Accelerated K-means with multiple random initializations,
        and picking one with the highest
    '''
    print('--------------Accelerated K-means--------------')
    tic = time.process_time()
    accKMeans_ = accKMeans(num_clusters=3, max_iterations=100, random_seed=10)
    accKMeans_.fit(X)
    toc = time.process_time()
    add_one = np.array(np.ones(X.shape[0]))
    accKMeans_result = np.int64(accKMeans_.assignments + add_one)
    
    correct_cnt = (accKMeans_result == y).sum()
    print('----Accuracy = {:.2f}%.'.format(correct_cnt / X.shape[0] * 100))
    print('----Rand index = ', rand_index(y, accKMeans_result))
    print('----Silhouette coefficient = ', silhouette_coef(X, accKMeans_result))
    print('----Number of iterations = ', accKMeans_.iteration_cnt)
    print ("----Computation time = " + str(1000 * (toc - tic)) + "ms")
    print('----Predicted class labels:\n', accKMeans_result)
    print('-----------------------------------------------')
    
    print('-------------------Sensitivity Analysis-------------------')
    sensitive_analysis(X, y, repeat_times=5, model='accKmeans')
    print('----------------------------------------------------')