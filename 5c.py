import numpy as np
import matplotlib.pyplot as plt

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

from surprise import SVD
from surprise import accuracy




def main():
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
    reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1, 5))
    Y1_train = Dataset.load_from_file('data/train.txt', reader=reader)
    trainset = Y1_train.build_full_trainset()
        
    algo = SVD(n_factors=20, n_epochs=50, lr_all=0.015, reg_all=0.1,verbose=True)
        
    algo.fit(trainset)    
    
    predictions = algo.test(Y_test)
    accuracy.rmse(predictions)
    
    predictions1 = algo.test(Y_train)
    accuracy.rmse(predictions1)

if __name__ == "__main__":
    main()
