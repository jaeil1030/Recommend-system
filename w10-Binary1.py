# Created on Oct 31 2019
# @author: 임일
# Binary data

import pandas as pd
import numpy as np
import random

# Data 읽어오기
articles = pd.read_csv('C:/RecoSys/Data/shared_articles.csv')
interactions = pd.read_csv('C:/RecoSys/Data/users_interactions.csv')
articles.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
interactions.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)

# Data merge
articles = articles[articles['eventType'] == 'CONTENT SHARED']
articles.drop('eventType', axis=1, inplace=True)
data = pd.merge(interactions[['contentId','personId', 'eventType']], articles[['contentId', 'title']], how='inner', on='contentId')

# Event 종류별로 다른 가중치 부여
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 1.0, 
   'FOLLOW': 2.0,
   'COMMENT CREATED': 2.0,  
}
data['eventStrength'] = data['eventType'].apply(lambda x: event_type_strength[x])

# 중복값 지우기
data = data.drop_duplicates()
grouped_data = data.groupby(['personId', 'contentId', 'title']).sum().reset_index()

# 데이터 값 recoding 하기 
grouped_data['personId'] = grouped_data['personId'].astype("category")
grouped_data['contentId'] = grouped_data['contentId'].astype("category")
grouped_data['person_id'] = grouped_data['personId'].cat.codes
grouped_data['content_id'] = grouped_data['contentId'].cat.codes

# Full rating matrix 구하기 
rating_matrix = np.array(grouped_data.pivot(index = 'person_id', columns ='content_id', values = 'eventStrength').fillna(0))

# New MF class for training & testing
class NEW_MF():
    # Initializing the object
    def __init__(self, rating_matrix, K, alpha, beta, iterations, verbose=True):
        self.R = rating_matrix
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
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

    def test(self):                             # Training 하면서 test set의 정확도를 계산하는 메소드 
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
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %4f" % (i+1, rmse1, rmse2))
        return training_process

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

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

    # Test RMSE 계산하는 method 
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.full_matrix[one_set[0], one_set[1]]
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)

# Testing MF RMSE
R_temp = rating_matrix.copy()               # Save original data
mf = NEW_MF(R_temp, K=105, alpha=0.001, beta=0.014, iterations=100, verbose=True)
test_set = mf.set_test(test_size=0.25)
result = mf.test()

