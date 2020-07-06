# Created on Oct 23 2019
# @author: 임일
# Surprise 1

import numpy as np
import pandas as pd
# Importing algorithms from Surprise
from surprise import SVD
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVDpp
from surprise import NMF
from surprise import BaselineOnly
from surprise import SlopeOne
# Importing other modules from Surprise
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
# Importing built in MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')

# Baseline 알고리즘 지정
algo = BaselineOnly()
# cv=4는 데이터를 4개로 나누어서 하나를 test set으로 사용하는데 5개 모두에 대해서 실행
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=4, verbose=True)

# Set full train data 지정, 예측하기 
trainset = data.build_full_trainset()
pred = algo.predict('1', '2', r_ui=3, verbose=True)  # user_id, item_id, default rating

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=4, verbose=True)

# Train/Test 분리 계산 
trainset, testset = train_test_split(data, test_size=0.25)
algo = KNNWithMeans()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# 알고리즘 옵션 변경
trainset, testset = train_test_split(data, test_size=0.25)
sim_options = {'name': 'pearson',
               'user_based': True
               }
algo = KNNWithMeans(sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Neighbor size 변경
trainset, testset = train_test_split(data, test_size=0.25)
sim_options = {'name': 'pearson',
               'user_based': True
               }
algo = KNNWithMeans(k=20, sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# 다양한 Neighbor size 비교 
trainset, testset = train_test_split(data, test_size=0.25)
for neighbor_size in (10, 20, 30, 40, 50, 60):
    algo = KNNWithMeans(k=neighbor_size, sim_options={'name': 'pearson_baseline', 'user_based': True})
    algo.fit(trainset)
    predictions = algo.test(testset)
    print('K = ', neighbor_size, 'RMSE = ', accuracy.rmse(predictions, verbose=False))
