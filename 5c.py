import numpy as np
import matplotlib.pyplot as plt

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from surprise import SVD
from surprise import accuracy


#trainset, testset = train_test_split(Y1_train, test_size=.25)    
#Y_train = np.loadtxt('data/train.txt').astype(int)
#Y_test = np.loadtxt('data/test.txt').astype(int)

def main():
    
    reader = Reader(line_format='user item rating', sep='\t',rating_scale=(1, 5))
    Y1_train = Dataset.load_from_file('data/train.txt', reader=reader)
    trainset = Y1_train.build_full_trainset()
    Y1_test = Dataset.load_from_file('./data/test.txt', reader=reader)
    Y_test = Y1_test.build_full_trainset()
    testset = Y_test.build_testset()
        
    algo = SVD(n_factors=50, n_epochs=100, lr_all=0.15, reg_all=0.1,verbose=True)
        
    algo.fit(trainset)    
    
    predictions = algo.test(testset)
    accuracy.rmse(predictions)
    
    #predictions1 = algo.test(trainset)
    #accuracy.rmse(predictions1)

if __name__ == "__main__":
    main()
