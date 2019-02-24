# For 5.2 add bias
# Authors: Hao Liu

import numpy as np
import random

def grad_U(Ui, Yij, Vj,ai, bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    grad = reg*Ui - Vj*(Yij-(np.dot(Ui,Vj)+ai+bj) )
    return eta*grad

def grad_V(Vj, Yij, Ui, ai, bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    grad = reg*Vj - Ui*(Yij-(np.dot(Ui,Vj)+ai+bj) )
    return eta*grad

def grad_A(Vj, Yij, Ui, ai, bj, reg, eta):
    grad =  - (Yij-(np.dot(Ui,Vj)+ai+bj) )
    return eta*grad



def grad_B(Vj, Yij, Ui, ai, bj, reg, eta):
    grad =  - (Yij-(np.dot(Ui,Vj)+ai+bj) )
    return eta*grad



def get_err(U, V, Y,A,B, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    fir = 0.5*reg*(np.linalg.norm(U,'fro')+np.linalg.norm(V,'fro'))
    sec = 0
    for ind in range(len(Y)):
        i,j,y_ij = Y[ind][0]-1,Y[ind][1]-1,Y[ind][2]
        sec = sec + (y_ij - (np.dot(U[i],V[j])+A[i] +B[j]  )        )**2
    return (fir + 0.5*sec)/len(Y)


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    Y = np.array(Y)
    print(Y.shape)
    U = np.random.uniform(-0.5, 0.5, (M, K)) #
    V = np.random.uniform(-0.5, 0.5, (N, K))
    A = np.random.uniform(-0.5, 0.5, (M, 1)) # bias term for each user
    B = np.random.uniform(-0.5, 0.5, (N, 1)) # bias term for each movie

    ep = 1
    prev_err = get_err(U,V,Y,A,B,reg)
    init_delta = 0
    curr_err = 0
    while ep <= max_epochs:
        #permute the data

        iter_ind = np.random.permutation(Y.shape[0])
        #gradient descend for each point
        for ind in iter_ind:
            i,j,y_ij = Y[ind][0]-1,Y[ind][1]-1,Y[ind][2]
            gu = grad_U(U[i],y_ij,V[j],A[i], B[j], reg,eta)
            gv = grad_V(V[j],y_ij,U[i],A[i], B[j], reg,eta)
            ga = grad_A(V[j],y_ij,U[i],A[i], B[j], reg,eta)
            gb = grad_B(V[j],y_ij,U[i],A[i], B[j], reg,eta)
            U[i] = U[i] - gu
            V[j] = V[j] - gv
            A[i] = A[i] - ga
            B[j] = B[j] - gb
        curr_err = get_err(U,V,Y,A,B,reg)
        print('@'+str(ep)+':'+str(curr_err))
        if ep == 1:
            init_delta = curr_err - prev_err
        else:
            if (curr_err - prev_err)/init_delta < eps:
                break
        prev_err = curr_err
        ep += 1
    return U,V,A,B,curr_err
