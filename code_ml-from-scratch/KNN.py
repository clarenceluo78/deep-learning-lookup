import scipy.io as sio
import numpy as np

atnt_X_train = sio.loadmat('./ATNT face/trainX.mat')['trainX']
atnt_Y_train = sio.loadmat('./ATNT face/trainY.mat')['trainY']
atnt_X_test = sio.loadmat('./ATNT face/testX.mat')['testX']
atnt_Y_test = sio.loadmat('./ATNT face/testY.mat')['testY']

binalpha_X_train = sio.loadmat('./Binalpha handwritten/trainX.mat')['trainX']
binalpha_Y_train = sio.loadmat('./Binalpha handwritten/trainY.mat')['trainY']
binalpha_X_test = sio.loadmat('./Binalpha handwritten/testX.mat')['testX']
binalpha_Y_test = sio.loadmat('./Binalpha handwritten/testY.mat')['testY']

########################################
# Define the class of KNN
########################################
class KNearestNeighbor():
    
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def compute_distance(self, X_test):
        
        X_test_squared = np.sum(np.transpose(np.square(X_test)),axis = 1).reshape(X_test.shape[1], 1)
        X_train_squared = np.sum(np.square(self.X_train), axis=0).reshape(1, self.X_train.shape[1])
        two_X_test_X_train = 2 * np.dot(np.transpose(X_test), self.X_train)
        return np.sqrt(X_test_squared + X_train_squared - two_X_test_X_train)
        
    def predict_labels(self, distances):
        
        y_pred = []
        sort_distances = np.sort(distances, axis = 1)
        
        for i in range(distances.shape[0]):
            dis_dict = {}
            label_list = []
            for j in range(distances.shape[1]):
                if distances[i][j] <= sort_distances[i][self.k-1]: #set value k
                    label_list.append(self.y_train[0][j])
            for k in range(len(label_list)):
                dis_dict[label_list[k]] = dis_dict.get(label_list[k], 0) + 1
            labels = sorted(dis_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
            y_pred.append(labels)
            
        return np.array(y_pred)
            
    def predict(self, X_test):
        distances = self.compute_distance(X_test)
        return self.predict_labels(distances)
     
     
# ATNT prediction
KNN = KNearestNeighbor(k=3)
KNN.train(atnt_X_train, atnt_Y_train)
y_pred = KNN.predict(atnt_X_test)
Accuracy = (y_pred == atnt_Y_test).sum() / atnt_Y_test.shape[1]
print('Accuracy: {:.2f}%.'.format(Accuracy*100))
print('The predicted class label: \n', y_pred)

# Handwritten prediction
KNN = KNearestNeighbor(k=3)
KNN.train(binalpha_X_train, binalpha_Y_train)
y_pred = KNN.predict(binalpha_X_test)
Accuracy = (y_pred == binalpha_Y_test).sum() / binalpha_Y_test.shape[1]
print('Accuracy: {:.2f}%.'.format(Accuracy*100))
print('The predicted class label: \n', y_pred)


