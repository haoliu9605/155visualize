import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils_5B import train_model, get_err

best_reg = 0.1
best_eta = 0.015

Y_train = np.loadtxt('data/train.txt').astype(int)
Y_test = np.loadtxt('data/test.txt').astype(int)

M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
print("Factorizing with ", M, " users, ", N, " movies.")
K = 20

# Perform CV when best param is not known
if best_reg == -1 and best_eta == -1:
    reg = [0.05, 0.1, 0.2, 0.4, 1]
    eta = [0.015, 0.03, 0.06] # learning rate
    E_out = np.zeros((len(reg),len(eta)))

    # Use to compute Ein and Eout
    for r in range(len(reg)):
        for e in range(len(eta)):
            U, V, A, B, err = train_model(M, N, K, eta[e], reg[r], Y_train)
            err_out = get_err(U, V, Y_test, A, B)
            E_out[r, e] = err_out
            print(r, e, err_out)

            if (best_reg == -1 and best_eta == -1) or E_out[best_reg, best_eta] > err_out:
                best_reg, best_eta = r, e

    data = pd.DataFrame(E_out, index = np.array(reg), columns = np.array(eta))
    sns.set()
    ax = sns.heatmap(data,cmap="YlGnBu",annot=True,fmt='.3g')
    fig = ax.get_figure()
    fig.savefig('heatmap_5B.png')

U, V, A, B, err = train_model(M, N, K, best_eta, best_reg, Y_train)
normV = V - np.mean(V, axis=0)

U = U.transpose()
normV = normV.transpose()

u, s, vh = np.linalg.svd(normV)
proj = u[:, :2]
lowdimU = np.matmul(proj.transpose(), U).transpose()
lowdimV = np.matmul(proj.transpose(), normV).transpose()

repreU = (lowdimU - np.mean(lowdimU, axis=0)) / np.std(lowdimU, axis=0)
repreV = (lowdimV - np.mean(lowdimV, axis=0)) / np.std(lowdimV, axis=0)