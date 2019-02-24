# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from utils_5B_adv import train_model, get_err

def main():
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    print("Factorizing with ", M, " users, ", N, " movies.")
    Ks = [10,20,30,50,100]

    reg = 0.0
    eta = 0.03 # learning rate
    E_in = []
    E_out = []
    
    #normalize / -mean
    Ylen = Y_train.shape[0]
    ind = list(range(Ylen))
    Ymean = np.mean(Y_train[:, 2])
    for t in ind:
        Y_train[t][2] = Y_train[t][2]-Ymean
    for t in list(range(Y_test.shape[0])):
        Y_test[t][2] = Y_test[t][2]-Ymean
    print(Ymean)
    
    # Use to compute Ein and Eout
    for K in Ks:
        U,V,A,B,err = train_model(M, N, K, eta, reg, Y_train,max_epochs=5)
        E_in.append(err)
        E_out.append(get_err(U, V, Y_test, A, B ))

    plt.plot(Ks, E_in, label='$E_{in}$')
    plt.plot(Ks, E_out, label='$E_{out}$')
    plt.title('Error vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('2d.png')



if __name__ == "__main__":
    main()
