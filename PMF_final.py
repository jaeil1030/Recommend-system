# Load library
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from preprocessing import ratings_features, users_preprocessing, books_preprocessing #직접 제작한 전처리 함수
from numpy.random import RandomState
import pickle
import os
import copy
from sklearn.model_selection import train_test_split


# Load data
ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='latin-1')
books = pd.read_csv('BX-Books.csv', encoding='latin-1')
users = pd.read_csv('BX-Users.csv', encoding='latin-1')

# Rating이 0인 열 삭제
non_zero_ratings = ratings.loc[ratings['Book-Rating'] != 0]

# ISBN과 User-ID Reindexing
isbn = pd.DataFrame(non_zero_ratings["ISBN"].unique().tolist(), columns=["ISBN"])
userid = pd.DataFrame(non_zero_ratings["User-ID"].unique().tolist(), columns=["User-ID"])
temp = pd.DataFrame(np.arange(0,44133).tolist(), columns=["ISBN2"])
temp2 = pd.DataFrame(np.arange(0,5157).tolist(), columns=["User-ID2"])

# Data merge
isbn2 = pd.concat([isbn, temp], axis=1)
userid2 = pd.concat([userid, temp2], axis=1)
ratings = non_zero_ratings.merge(isbn2, how='left')
ratings = ratings.merge(userid2, how='left')
ratings = ratings.drop(['User-ID', 'ISBN'], axis=1)
ratings = ratings[['User-ID2', 'ISBN2', 'Book-Rating']]
ratings.columns = ['user_id', 'ISBN', 'rating']


# Probabilistic Matrix Factorization (PMF)
NUM_USERS = int(max(ratings.user_id + 1))
NUM_ITEMS = int(max(ratings.ISBN + 1))
print('Max user:{}, Max item:{}'.format(NUM_USERS, NUM_ITEMS))
print('dataset density: {:f}'.format(len(ratings)*1.0/(NUM_USERS*NUM_ITEMS)))

# Rating matrix 생성  (Ordered)
R = np.zeros([NUM_USERS, NUM_ITEMS])
for idx in range(len(ratings)):
    user_idx = ratings.iloc[idx,0]
    book_idx = ratings.iloc[idx,1]
    rating_ = ratings.iloc[idx,2]
    R[user_idx, book_idx] = rating_

# Rating Matrix shape 확인
print('R matrix shape:', R.shape)

# Train, test set split
train, test = train_test_split(ratings, test_size=0.3, random_state=42)

# 평점에서 데이터별 평균 빼주기
train.loc[:, 'adj_rating'] = train['rating'].apply(lambda x: x - train['rating'].mean())
train_new = train[['user_id', 'ISBN', 'adj_rating']].copy()
test.loc[:, 'adj_rating'] = test['rating'].apply(lambda x: x - test['rating'].mean())
test_new = test[['user_id', 'ISBN', 'adj_rating']].copy()

# As.array: train test split
train_new = np.array(train_new, dtype='int64')
test_new = np.array(test_new, dtype='int64')

# PMF Class 정의
class PMF():
    '''
    a class for this Double Co-occurence Factorization model
    '''
    # initialize some paprameters
    def __init__(self, R, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=100, momuntum=0.5,
                 lr=0.001, iters=1300, seed=None):
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.momuntum = momuntum
        self.R = R
        self.random_state = RandomState(seed)
        self.iterations = iters
        self.lr = lr
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1
        self.U = 0.1*self.random_state.rand(np.size(R, 0), latent_size)
        self.V = 0.1*self.random_state.rand(np.size(R, 1), latent_size)

    def loss(self):
        # the loss function of the model
        loss = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2) + self.lambda_alpha*np.sum(np.square(self.U)) + self.lambda_beta*np.sum(np.square(self.V))
        return loss
    
    def RMSE(self, preds, truth):
        '''RMSE'''
        return np.sqrt(np.mean(np.square(preds-truth)))

    def train(self, train_data=None, vali_data=None, verbose=True):
        '''
        # training process
        :param train_data: train data with [[i,j],...] and this indacates that K[i,j]=rating
        :param lr: learning rate
        :param iterations: number of iterations
        :return: learned V, T and loss_list during iterations
        '''
        # histroy 
        train_loss_list = []
        trian_rmse = []
        vali_rmse_list = []
        last_vali_rmse = None

        # monemtum
        momuntum_u = np.zeros(self.U.shape)
        momuntum_v = np.zeros(self.V.shape)

        patience = 0
        for it in range(self.iterations):

            # derivate of Vi
            grads_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha*self.U

            # derivate of Tj
            grads_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta*self.V

            # update the parameters
            momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
            momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
            self.U = self.U - momuntum_u
            self.V = self.V - momuntum_v

            # training evaluation
            train_loss = self.loss()
            train_loss_list.append(train_loss)

            vali_preds = self.predict(vali_data)
            vali_rmse = self.RMSE(vali_data[:,2].ravel(), vali_preds.ravel())
            vali_rmse_list.append(vali_rmse)

            print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, vali_rmse))
            
            
            if last_vali_rmse and (last_vali_rmse - vali_rmse) <= 0:
                patience += 1
                print('convergence at iterations:{: d}, patience:{}'.format(it, patience))
                if patience >= 3:
                    break # RMSE가 3번 이상 상승하면 학습 종료
            else:
                last_vali_rmse = vali_rmse

        return self.U, self.V, train_loss_list, vali_rmse_list
    
    def predict(self, data):
        index_data = np.array([[int(ele[0]), int(ele[1])] for ele in data], dtype='int32')
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        preds_value_array = np.sum(u_features*v_features, 1)
        return preds_value_array

# PMF 하이퍼파라미터 설정
pmf = PMF(R=R,
           lambda_alpha = 0.001, 
           lambda_beta = 0.01, 
           latent_size = 30, 
           momuntum = 0.07,
           lr=0.00015, 
           iters=100)

# PMF 실행 후 결과 확인
U, V, train_loss, valid_rmse = pmf.train(train_data=train_new, vali_data=test_new, verbose=True)

