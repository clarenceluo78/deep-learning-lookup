import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def MDS(D):
    D2 = D**2
    assert(D.shape == (10, 10) and D2.shape == (10, 10))
    print("----The squared proximity matrix D2:\n", D2)
    
    num_cities = D.shape[0]
    I = np.eye(num_cities)
    J = np.ones((num_cities, num_cities))
    C = I - 1/num_cities*J
    B = -(1/2) * np.dot(np.dot(C, D), C)
    assert(B.shape == (10, 10))
    print("----The centered matrix B:\n", B)
    
    U, S, Vt = scipy.linalg.svd(B)
    print("----The eigenvectors of B:\n", Vt)
    eigenvalues = S[0:2]
    Lambda = np.diag(eigenvalues)
    E = U[:, 0:2]
    print("----The eigenvectors correspond to the two largest eigenvalues E:\n", E)
    X = np.dot(E, np.sqrt(Lambda))*100  # denote in hundreds of miles
    print("----The embedding coordinates X:\n", X)
    
    return X, num_cities

def plot(positions, num_cities):
    # # 1st embedding
    # x_data = positions[:, 0]
    # y_data = positions[:, 1]
    
    # # 2nd embedding
    # positions[:, 1] = -positions[:, 1]
    # x_data = positions[:, 0]
    # y_data = positions[:, 1]
    
    # 3rd embedding
    positions[:, 0] = -positions[:, 0]
    positions[:, 1] = -positions[:, 1]
    x_data = positions[:, 0]
    y_data = positions[:, 1]
    
    x = MultipleLocator(100)
    y = MultipleLocator(100) 
    ax = plt.gca()
    ax.xaxis.set_major_locator(x)
    ax.yaxis.set_major_locator(y)

    plt.scatter(x_data, y_data)

    for i in range(num_cities):
        plt.annotate(city_names[i], positions[i])
        
    plt.show()


if __name__ == '__main__':
    city_names = [
        "Madrid", 
        "Vancouver",
        "Napoli",
        "Kelowna",
        "San Francisco",
        "Palermo",
        "Seattle",
        "Shanghai",
        "Sydney",
        "Tokyo"]
    D = np.loadtxt("ha30_dist.txt")
    D = D[20:, 20:]
    print("----The distance matrix D:\n", D)

    positions, num_cities = MDS(D)
    plot(positions, num_cities)
    