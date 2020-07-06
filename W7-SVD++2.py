# Created on Sun Oct 13 2019
# @author: 임일
# SVD++

import numpy as np
import pandas as pd
import random

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
rating_matrix = np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))

class SVD_PP():
    # Initializing the object
    def __init__(self, rating_matrix, K, alpha, beta, lamda, iterations, tolerance=0.005, verbose=True):
        self.R = rating_matrix
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.iterations = iterations
        self.tolerance = tolerance
        self.best_RMSE = 10000
        self.verbose = verbose

    # 테스트 셋을 선정하는 메소드 
    def set_test(self, test_size=0.25):         # Setting test set
        xs, ys = self.R.nonzero()
        test_set = []
        for x, y in zip(xs, ys):                # Random selection
            if random.random() < test_size:
                test_set.append([x,y,self.R[x,y]])
                self.R[x,y] = 0
        self.test_set = test_set
        return test_set                         # Return test set

    def test(self):
        # Initializing user-feature and movie-feature matrix 
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.y = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # 평가한 영화를 1로 바꾼 implicit data 생성
        self.Ru = (self.R > 0).astype(float)
        self.Ru_1_2 = np.sqrt(np.sum(self.Ru, axis=1))

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            self.full_matrix = self.full_prediction()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %4f" % (i+1, rmse1, rmse2))
            if self.best_RMSE > rmse2:                      # New best record
                self.best_RMSE = rmse2
            elif (rmse2 - self.best_RMSE) > self.tolerance: # RMSE is increasing over tolerance
                break
        return training_process

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.lamda * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * (self.P[i, :] + np.dot(self.Ru[i,:], self.y) / self.Ru_1_2[i]) - self.lamda * self.Q[j,:])
            self.y[j, :] += self.alpha * (e * self.Q[j,:] / self.Ru_1_2[i] - self.lamda * self.y[j, :])

    # Computing mean squared error
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        error = 0
        for x, y in zip(xs, ys):
            self.predictions.append(self.full_matrix[x, y])
            self.errors.append(self.R[x, y] - self.full_matrix[x, y])
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.full_matrix[one_set[0], one_set[1]]
            if predicted > 5:
                predicted = 5
            if predicted < 1:
                predicted = 1            
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))
    
    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + (self.P[i, :] + np.dot(self.Ru[i, :], self.y) / self.Ru_1_2[i]).dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis,:] + (self.P + np.dot(self.Ru, self.y) / self.Ru_1_2[:, np.newaxis]).dot(self.Q.T)


# Testing SVD RMSE
R_temp = rating_matrix.copy()               # Save original data
SVD = SVD_PP(R_temp, K=90, alpha=0.001, beta=0.01, lamda=0.0003, iterations=300, tolerance=0.005, verbose=True)
test_set = SVD.set_test(test_size=0.25)
result = SVD.test()



# To find optimal K
results = []
index = []
for K in range(90, 111, 3):
    print('K =', K)
    R_temp = rating_matrix.copy()               # Save original data
    SVD = SVD_PP(R_temp, K=K, alpha=0.001, beta=0.01, lamda=0.0003, iterations=300, tolerance=0.005, verbose=True)
    test_set = SVD.set_test(test_size=0.25)
    result = SVD.test()
    index.append(K)
    results.append(result)

summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    min = np.min(RMSE)
    j = RMSE.index(min)
    summary.append([index[i], j+1, RMSE[j]])

import matplotlib.pyplot as plt
plt.plot(index, [x[2] for x in summary])
plt.ylim(0.875, 0.95)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()



# To find optimal alpha
results = []
index = []
for i in range(7, 15):
    alpha = i/10000
    print('alpha =', alpha)
    R_temp = rating_matrix.copy()               # Save original data
    SVD = SVD_PP(R_temp, K=105, alpha=alpha, beta=0.01, lamda=0.0001 iterations=300, tolerance=0.005, verbose=True)
    test_set = SVD.set_test(test_size=0.25)
    result = SVD.test()
    index.append(alpha)
    results.append(result)

summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    min = np.min(RMSE)
    j = RMSE.index(min)
    summary.append([index[i], j+1, RMSE[j]])

import matplotlib.pyplot as plt
plt.plot(index, [x[2] for x in summary])
plt.ylim(0.875, 0.95)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.show()



    
# To find optimal beta
results = []
index = []
for i in range(8, 19):
    beta = i/1000
    print('beta =', beta)
    R_temp = rating_matrix.copy()               # Save original data
    SVD = SVD_PP(R_temp, K=105, alpha=0.001, beta=beta, iterations=300, tolerance=0.005, verbose=True)
    test_set = SVD.set_test(test_size=0.25)
    result = SVD.test()
    index.append(beta)
    results.append(result)

summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    min = np.min(RMSE)
    j = RMSE.index(min)
    summary.append([index[i], j+1, RMSE[j]])

import matplotlib.pyplot as plt
plt.plot(index, [x[2] for x in summary])
plt.ylim(0.875, 0.95)
plt.xlabel('Beta')
plt.ylabel('RMSE')
plt.show()




# To find optimal lamda
results = []
index = []
for i in range(1,15):
    lamda = i/10000
    print('lamda =', lamda)
    R_temp = rating_matrix.copy()               # Save original data
    SVD = SVD_PP(R_temp, K=105, alpha=0.001, beta=0.01, lamda=lamda, iterations=300, tolerance=0.005, verbose=True)
    test_set = SVD.set_test(test_size=0.25)
    result = SVD.test()
    index.append(lamda)
    results.append(result)

summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    min = np.min(RMSE)
    j = RMSE.index(min)
    summary.append([index[i], j+1, RMSE[j]])

import matplotlib.pyplot as plt
plt.plot(index, [x[2] for x in summary])
plt.ylim(0.875, 0.95)
plt.xlabel('Lamda')
plt.ylabel('RMSE')
plt.show()

