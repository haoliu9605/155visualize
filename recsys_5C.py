import numpy as np
import matplotlib.pyplot as plt

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV


from surprise import SVD, SVDpp
from surprise import accuracy

# to_inner_uid(), to_inner_iid(), to_raw_uid(), and to_raw_iid()


def getUV():

    reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1, 5))

    Y_train = Dataset.load_from_file('data/train.txt', reader=reader)
    trainset = Y_train.build_full_trainset()
    train_perf = trainset.build_testset()

    Y_test = Dataset.load_from_file('./data/test.txt', reader=reader)
    ytest = Y_test.build_full_trainset()
    testset = ytest.build_testset()

    Y_data = Dataset.load_from_file('./data/data.txt', reader=reader)
    train_data = Y_data.build_full_trainset()

    test_data = train_data.build_testset()
    '''
    # grid search
    param_grid = {'n_factors':[20], 'n_epochs': [20], 'lr_all': [0.01,0.03, 0.05],
              'reg_all': [0.1, 0.2,0.3]}

    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
    gs.fit(Y_train)
    # best RMSE score
    print(0.5*gs.best_score['rmse']**2)
    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])
    '''

    # see performance on testset
    algo1 = SVD(n_factors=20, n_epochs=35, lr_all=0.03, reg_all=0.1,verbose=True)
    algo1.fit(trainset)

    predictions2 = algo1.test(train_perf)
    print("train ",0.5*accuracy.rmse(predictions2)**2) # rescale as the hw error

    predictions1 = algo1.test(testset)
    print("test ",0.5*accuracy.rmse(predictions1)**2) # rescale as the hw error

    # see visualization
    algo = SVD(n_factors=20, n_epochs=35, lr_all=0.03, reg_all=0.1,verbose=True)
    algo.fit(train_data)

    predictions = algo.test(test_data)
    print("data ",0.5*accuracy.rmse(predictions)**2) # rescale as the hw error

    U = algo.pu    # ndarray of size (user x factor)
    V = algo.qi    # ndarray of size (item x factor)
    
    
    # remapping of U, V:
    #U_ind = [] # U_ind[i] = j, then now ith row of U should be map to jth row of the new mat
    #for i in range(len(U)):
    #    U_ind.append(train_data.to_raw_iid(i))
    #V_ind = []    
    #for i in range(len(V)):
    #   V_ind.append(train_data.to_raw_uid(i))
    U_vec = range(len(U))
    U[[int(train_data.to_raw_uid(i))-1 for i in U_vec]] = U[U_vec]   
    V_vec = range(len(V))
    V[[int(train_data.to_raw_iid(i))-1 for i in V_vec]] = V[V_vec]
    return U,V
