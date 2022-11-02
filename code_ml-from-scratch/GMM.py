import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import random

from .metrics import rand_index, silhouette_coef, sensitive_analysis


class GMM():
    """GMM clustering
    """
    def __init__(self, num_clusters=3, tolerance=1e-3, random_seed=10, show_epochs=True):
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.tolerance = tolerance
        self.show_epochs = show_epochs
        self.X = None
        self.num_features = None
        self.num_samples = None
        self.assignments = None
        self.iteration_cnt = 0
        self.means = None                   # means 
        self.covariances = None             # covariance
        self.pi = None                      # mixing coefficients pi(s)
        self.gammas = None                  # gammas
      
    def init_parameters(self):
        np.random.seed(self.random_seed)
        
        # init evenly distributed fit_Gaussian coefficient pi
        self.pi = [1/self.num_clusters for i in range(self.num_clusters)]

        # init covariance as identity matrixes
        Var_k = np.eye(self.num_features)
        self.covariances = [Var_k[:] for i in range(0, self.num_clusters)]
        
        # init means
        self.means = np.ones((self.num_clusters, self.num_features))
        centroids = np.zeros((self.num_clusters, self.num_features))
        for k in range(self.num_clusters):
            init_idx = np.random.choice(range(self.num_samples))
            centroids[k] = X[init_idx]
            # print(init_idx)
        self.means = centroids
        if (self.show_epochs):
            print("Choose initial mean as:\n", centroids)

    def calculate_distance(self, a, b):
        
        length_square = [(a[i] - b[i])**2 for i in range(0, len(a))]
        distance = pow(sum(length_square), 0.5)
        # distance = np.linalg.norm(np.array([p1 - p2]))

        return distance
 
    def fit_Gaussian(self, x, mu, var):
        """fit Gaussian distribution with parameters

        Args:
            x (_type_): data point
            mu (_type_): mean
            var (_type_): standard variance

        Returns:
            _type_: Gaussian distribution
        """
        former = 1 / ((2*np.pi)**(len(x)/2) * (np.linalg.det(var)**0.5))
        x_minus_mu = (np.array(x) - np.array(mu)).reshape(1,-1)
        latter = np.exp(np.matmul(np.matmul(x_minus_mu, np.linalg.inv(var)), np.transpose(x_minus_mu)) * (-0.5))
        
        distribution = former * latter

        return distribution
    
    def E_step(self):
        """E step of EM algorithm
        """
        # init gammas and cluster assignments
        Gaussian_part = np.zeros((self.num_samples, self.num_clusters))
        self.gammas = np.zeros((self.num_samples, self.num_clusters))
        self.assignments = [-1 for i in range(0, self.num_samples)]

        # calculate partial Gaussian
        for sample_idx in range(self.num_samples):
            for cluster_idx in range(self.num_clusters):
                Gaussian_part[sample_idx][cluster_idx] = self.pi[cluster_idx] *\
                    self.fit_Gaussian(self.X[sample_idx], self.means[cluster_idx], self.covariances[cluster_idx])

        # calculate gamma with partial Gaussian values
        for sample_idx in range(self.num_samples):
            for cluster_idx in range(0, self.num_clusters):
                self.gammas[sample_idx][cluster_idx] = Gaussian_part[sample_idx][cluster_idx] / sum(Gaussian_part[sample_idx])

        # update assignments
        for sample_idx in range(self.num_samples):
            self.assignments[sample_idx] = self.gammas[sample_idx].argmax(axis=0)

    def M_step(self):
        """M step of EM algorithm
        """
        sum_gammas = self.gammas.sum(axis = 0)
        
        # for each cluster
        for cluster_idx in range(self.num_clusters):

            # update means
            sum_gamma_x = np.zeros(self.num_features)
            for n in range(self.num_samples):
                sum_gamma_x += self.gammas[n][cluster_idx] * np.array(self.X[n])
            self.means[cluster_idx] = sum_gamma_x / sum_gammas[cluster_idx]
            
            # update covariance
            sum_cov = np.zeros((self.num_features, self.num_features))
            for n in range(self.num_samples):
                x_minus_mu = (np.array(self.X[n]) - np.array(self.means[cluster_idx])).reshape(-1,1)
                sum_cov += self.gammas[n][cluster_idx] * np.matmul(x_minus_mu, x_minus_mu.T)
            self.covariances[cluster_idx] = sum_cov / sum_gammas[cluster_idx]
            
        # update pi
        self.pi = sum_gammas / self.num_samples

    def calculate_log_likelihood(self):
        log_likelihood = 0
        
        for sample_idx in range(self.num_samples):
            temp = 0
            for k in range(0, self.num_clusters):
                temp += self.pi[k] * self.fit_Gaussian(self.X[sample_idx], self.means[k], self.covariances[k])
            log_likelihood += np.log(temp)
            
        return log_likelihood 

    def fit(self, X):
        """Fit and train GMM model

        Args:
            X (_type_): data sample
        """
        self.X = X
        self.num_samples, self.num_features = X.shape
        old_log_likelihood = 0
        
        self.init_parameters()
        
        if (self.show_epochs):
            print("Start training:")
            print("Processing, please wait...")
        self.E_step()
        self.M_step()
        log_likelihood = self.calculate_log_likelihood()
        self.iteration_cnt += 1
        
        # repeat until convergence
        while (abs(log_likelihood - old_log_likelihood) >= self.tolerance):
            self.E_step()
            self.M_step()
            old_log_likelihood = log_likelihood
            log_likelihood = self.calculate_log_likelihood()
            
            self.iteration_cnt += 1
        
        if (self.show_epochs):
            print("GMM has converged!")

if __name__ == '__main__':
    seeds = np.loadtxt("seeds_dataset.txt")
    X = seeds[:, :7]
    y = np.int64(seeds[:, 7])
    
    ''' GMM:
    Runing GMM with multiple random initializations,
    and picking one with the highest
    '''
    print('--------------------GMM-EM--------------------')
    tic = time.process_time()
    GMM_ = GMM(num_clusters=3, tolerance=1e-3, random_seed=10)
    GMM_.fit(X)
    toc = time.process_time()
    add_one = np.array(np.ones(X.shape[0]))
    GMM_result = np.int64(GMM_.assignments + add_one)
    
    correct_cnt = (GMM_result == y).sum()
    print('----Accuracy = {:.2f}%.'.format(correct_cnt / X.shape[0] * 100))
    print('----Rand index = ', rand_index(y, GMM_result))
    print('----Silhouette coefficient = ', silhouette_coef(X, GMM_result))
    print('----Number of iterations = ', GMM_.iteration_cnt)
    print ("----Computation time = " + str(1000 * (toc - tic)) + "ms")
    print('----Predicted class labels:\n', GMM_result)
    print('-----------------------------------------------')
    
    print('-------------------Sensitivity Analysis-------------------')
    sensitive_analysis(X, y, repeat_times=5, model='GMM')
    print('----------------------------------------------------')