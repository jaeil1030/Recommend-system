# Created on Sun Oct 6 2019
# @author: 임일
# Matrix Factorization (MF) 1

import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
rating_matrix = np.array(ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))

class MF():
    # Initializing the object
    def __init__(self, rating_matrix, K, alpha, beta, iterations, verbose=True):
        self.R = rating_matrix
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

    def train(self):
        # Initializing user-feature and movie-feature matrix 
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

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
            measure = self.rmse()
            training_process.append((i, measure))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; RMSE = %.4f" % (i+1, measure))
        return training_process

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

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)


# Original MF
mf = MF(rating_matrix, K=30, alpha=0.001, beta=0.02, iterations=20, verbose=True)
mf.train()
print("Full matrix (P x Qt):")
print(mf.full_matrix)
print(mf.get_prediction(1,1))

