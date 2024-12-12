import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
import time
from mpl_toolkits.mplot3d import Axes3D

def driver():
    # THIS CODE IS FOR NEWTON'S ON STRICTLY CONVEX FUNCTION R6

    Nmax = 100
    x0 = np.array([1.9, 1.7, 0.8, 1.4, 0.6,1.3])
    tol = 1e-12

    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors, evals] = NewtonMethod(x0, tol, Nmax)
    end_time = time.perf_counter()
    print("xstar:", xstar)
    print("f(xstar):", eval_f(xstar))
    print("||", chr(8711), "F(xstar)||:", gval)
    print("ier:", ier)
    print("Time elapsed:", end_time - start_time)
    diff = np.array(evals-np.array([0,0,0,0,0,0]))
    norms = []
    for i in range(len(diff)):
        norms.append(norm(diff[i]))

    print("Iterations: ", norms)

    iterations = range(len(errors))
    convergence_order(errors[:-1], gval)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(iterations, np.abs(errors), color='dodgerblue', marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$e_k$', fontsize=14)
    plt.title("Semiology Error Plot of Newton's on $F(x)$", fontsize=16)
    plt.show()


def eval_f(x):
    return np.exp(x[0])-x[0] + np.exp(x[1])-x[1] + np.exp(x[2])-x[2] + np.exp(x[3])-x[3] + np.exp(x[4])-x[4] + np.exp(x[5])-x[5]

def evalF(x):
    F = np.zeros(6)
    F[0] = np.exp(x[0]) - 1
    F[1] = np.exp(x[1]) - 1
    F[2] = np.exp(x[2]) - 1
    F[3] = np.exp(x[3]) - 1
    F[4] = np.exp(x[4]) - 1
    F[5] = np.exp(x[5]) - 1
    return F

def evalJ(x):
    J = np.array([
        [np.exp(x[0]), 0, 0, 0, 0, 0],
        [0, np.exp(x[1]), 0, 0, 0, 0],
        [0, 0, np.exp(x[2]), 0, 0, 0],
        [0, 0, 0, np.exp(x[3]), 0, 0],
        [0, 0, 0, 0, np.exp(x[4]), 0],
        [0, 0, 0, 0, 0, np.exp(x[5])]
    ])
    return J

def eval_norm(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2 + F[3]**2 + F[4]**2 + F[5]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    return 2 * np.dot(J.T, F)

def eval_hessianf(x):
    F = evalF(x)
    J = evalJ(x)
    second_derivatives = np.zeros((len(x), len(x)))
    for i in range(len(F)):
        Fi_hessian = evalH(x, i)
        second_derivatives += 2 * F[i] * Fi_hessian
    return 2 * np.dot(J.T, J) + second_derivatives

def evalH(x, i):
    if i == 0:
        return np.array([
            [np.exp(x[0]), 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
    elif i == 1:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, np.exp(x[1]), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
    elif i == 2:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, np.exp(x[2]), 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
    elif i == 3:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, np.exp(x[3]), 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
    elif i == 4:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, np.exp(x[4]), 0],
            [0, 0, 0, 0, 0, 0]
        ])
    elif i == 5:
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, np.exp(x[5])]
        ])

def convergence_order(errors, gval):
    if len(errors) >3:
        errors = errors[-3:]
    diff1 = np.abs(errors[1:])
    diff2 = np.abs(errors[:-1])
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    #print('The order equation is:')
    #print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where:')
    #print('lambda =', str(np.exp(fit[1])))
    print(r'asymptotic convergence order:', chr(945),'=', str(fit[0]))
    return [fit, diff1, diff2]

def NewtonMethod(x, tol, Nmax):
    errors = []
    evals = []
    for its in range(Nmax):
        gradf = eval_gradg(x)
        H = eval_hessianf(x)
        evals.append(x.copy())

        try:
            delta = -np.linalg.solve(H, gradf)
        except np.linalg.LinAlgError:
            print("Singular Hessian matrix")
            ier = 1
            return [x, norm(gradf), ier, errors, evals]
        
        x += delta
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x, gval, ier, errors, evals]
    
    print('Max iterations exceeded')
    ier = 1
    return [x, gval, ier, errors, evals]



if __name__ == '__main__':
    driver()
