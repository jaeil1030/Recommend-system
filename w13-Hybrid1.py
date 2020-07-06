# Created on Sun Nov 24 2019
# @author: 임일
# Hybrid Recommender 1

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

# np.array로 바꾸기
ratings_train = np.array(ratings_train.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))
ratings_test = np.array(ratings_test)

# Dummy recommender 0
def recommender0(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random() * 4 + 1)
    return recommendations

# Dummy recommender 1
def recommender1(recomm_list):
    recommendations = []
    for pair in recomm_list:
        recommendations.append(random.random() * 4 + 1)
    return recommendations

# Hybrid 함수
def hybrid1(recomm_list, weight=[0.5, 0.5]):  
    result0 = recommender0(recomm_list)
    result1 = recommender1(recomm_list)
    result = []
    for i, number in enumerate(result0):
        result.append(result0[i] * weight[0] + result1[i] * weight[1])
    return result

# RMSE 계산을 위한 함수
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Hybrid 결과 얻기
predictions = hybrid1(ratings_test[:, [0, 1]], [0.8, 0.2])
RMSE2(ratings_test[:, 2], predictions)

