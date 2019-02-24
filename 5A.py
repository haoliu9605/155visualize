# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils_5A import train_model, get_err
def main():
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
	
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20
	
    reg = [0.00001, 0.0001, 0.001, 0.1,1]
    eps = [0.00001,0.0001,0.001,0.1,1]
    reg = [0.1,1]
    eps = [0.1,1]
    eta = 0.03 # learning rate
    E_out = np.zeros((len(reg),len(eps)))
    # Use to compute Ein and Eout
    for r in range(len(reg)):
        for e in range(len(eps)):
            U,V, err = train_model(M, N, K, eta, reg[r], Y_train,eps[e])
            err_out = get_err(U, V, Y_test)
            E_out[r,e] = err_out

    data = pd.DataFrame(E_out,index = np.array(reg), columns = np.array(eps))
    sns.set()
    ax = sns.heatmap(data,cmap="YlGnBu")
    fig = ax.get_figure()
    fig.savefig('heatmap_5A.png')

   

if __name__ == "__main__":
    main()
