import numpy as np
import matplotlib.pyplot as plt

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV


from surprise import SVD, SVDpp
from surprise import accuracy




def main():

    reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1, 5))

    Y_train = Dataset.load_from_file('data/train.txt', reader=reader)
    trainset = Y_train.build_full_trainset()

    Y_test = Dataset.load_from_file('./data/test.txt', reader=reader)
    ytest = Y_test.build_full_trainset()
    testset = ytest.build_testset()

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
    algo = SVD(n_factors=20, n_epochs=20, lr_all=0.03, reg_all=0.1,verbose=True)
    algo.fit(trainset)

    predictions = algo.test(testset)
    print(0.5*accuracy.rmse(predictions)**2) # rescale as the hw error

    U = np.transpose(algo.pu)    # ndarray of size (user x factor)
    V = np.transpose(algo.qi)    # ndarray of size (item x factor)

    '''

if __name__ == "__main__":
    main()
