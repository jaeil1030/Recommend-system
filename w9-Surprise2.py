# Created on Oct 23 2019
# @author: 임일
# Surprise 2

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

# 알고리즘 비교
trainset, testset = train_test_split(data, test_size=0.25)
algorithms = [KNNWithMeans, KNNWithZScore, SVD, SVDpp, NMF, SlopeOne]
# 결과를 저장할 변수 
names = []
results = []
# Loop 
for option in algorithms:
    algo = option()
    names.append(option.__name__)       # 알고리즘 이름 
    algo.fit(trainset)
    predictions = algo.test(testset)
    results.append(accuracy.rmse(predictions))
names = np.array(names)
results = np.array(results)

# 결과를 그래프로 표시
import matplotlib.pyplot as plt

index = np.argsort(results)
plt.ylim(0.8,1)
plt.plot(names[index], results[index])

