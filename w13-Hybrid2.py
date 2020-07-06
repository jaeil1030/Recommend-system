# Created on Sun Nov 24 2019
# @author: 임일
# Hybrid Recommender 2

import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

# train test 분리
TRAIN_SIZE = 0.75
ratings = shuffle(ratings)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

ratings = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)

# Predictions using MF ###########################################################################
class NEW_MF():
    # Initializing the object
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings)
        # user_id, movie_id를 R의 index와 매칭하기 위한 dictionary 생성
        movie_id_index = []
        index_movie_id = []
        for i, one_id in enumerate(ratings):
            movie_id_index.append([one_id, i])
            index_movie_id.append([i, one_id])
        self.movie_id_index = dict(movie_id_index)
        self.index_movie_id = dict(index_movie_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

    # 테스트 셋을 선정하는 메소드 (Hybrid를 위해 수정)
    def set_test(self, ratings_test):                           # Setting test set
        test_set = []
        for i in range(len(ratings_test)):                      # Selected ratings
            x = self.user_id_index[ratings_test.iloc[i,0]]      # Getting R indice for the given user_id and movie_id
            y = self.movie_id_index[ratings_test.iloc[i,1]]
            z = ratings_test.iloc[i,2]
            test_set.append([x, y, z])
            self.R[x, y] = 0   # Setting test set ratings to 0
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
        self.full_matrix = self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)
        return self.full_matrix

    def get_one_prediction(self, user_id, movie_id):
        return self.full_matrix[self.user_id_index[user_id], self.movie_id_index[movie_id]]

# Creating MF Object & train
ratings_temp = ratings.copy()
mf = NEW_MF(ratings_temp, K=105, alpha=0.001, beta=0.014, iterations=100, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()


# Predictions using CF ###########################################################################
from sklearn.metrics.pairwise import cosine_similarity
rating_matrix = ratings_train.pivot_table(values='rating', index='user_id', columns='movie_id')

# 모든 가능한 사용자 pair의 Cosine similarities 계산
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산 
rating_mean = rating_matrix.mean(axis=1)

def ubcf_bias_knn(user_id, movie_id, neighbor_size=0):
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자 간의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 movie의 rating 가져오기
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 movie에 대한 rating이 없는 user 삭제
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        if neighbor_size == 0:               # Neighbor size가 지정되지 않은 경우
            # 편차로 예측치 계산
            movie_ratings = movie_ratings - others_mean
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            # 예측값에 현 사용자의 평균 더하기
            prediction = prediction + user_mean
        else:                                # Neighbor size가 지정된 경우
            # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
            neighbor_size = min(neighbor_size, len(sim_scores))
            # array로 바꾸기 (argsort를 사용하기 위함)
            sim_scores = np.array(sim_scores)
            movie_ratings = np.array(movie_ratings)
            others_mean = np.array(others_mean)
            # 유사도를 순서대로 정렬
            user_idx = np.argsort(sim_scores)
            # 유사도와 rating을 neighbor size만큼 받기
            sim_scores = sim_scores[user_idx][-neighbor_size:]
            movie_ratings = movie_ratings[user_idx][-neighbor_size:]
            # 사용자의 mean을 neighbor size만큼 받기
            others_mean = others_mean[user_idx][-neighbor_size:]
            # 편차로 예측치 계산
            movie_ratings = movie_ratings - others_mean
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            # 예측값에 현 사용자의 평균 더하기
            prediction = prediction + user_mean
    else:
        prediction = user_mean
    return prediction

# Hybrid recommendation ###########################################################################

def recommender0(recomm_list, mf):
    recommendations = []
    mf.full_prediction()
    for i in range(len(recomm_list)):
        recommendations.append(mf.get_one_prediction(recomm_list[i,0], recomm_list[i,1]))
    return np.array(recommendations)

def recommender1(recomm_list, neighbor_size=0):
    id_pairs = zip(recomm_list[:, 0], recomm_list[:, 1])
    recommendations = np.array([ubcf_bias_knn(user, movie, neighbor_size) for (user, movie) in id_pairs])
    return recommendations

# RMSE 계산을 위한 함수
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

recomm_list = np.array(ratings_test.iloc[:, [0, 1]])
result0 = recommender0(recomm_list, mf)
RMSE2(ratings_test.iloc[:, 2], result0)
result1 = recommender1(recomm_list, 37)
RMSE2(ratings_test.iloc[:, 2], result1)

weight = [0.8, 0.2]
predictions = []
for i, number in enumerate(result0):
    predictions.append(result0[i] * weight[0] + result1[i] * weight[1])
RMSE2(ratings_test.iloc[:, 2], predictions)





for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    weight = [i, 1-i]
    predictions = []
    for i, number in enumerate(result0):
        predictions.append(result0[i] * weight[0] + result1[i] * weight[1])
    print("Weights - %.1f : %.1f ; RMSE = %.5f" % (weight[0], weight[1], RMSE(ratings_test.iloc[:, 2], predictions)))
