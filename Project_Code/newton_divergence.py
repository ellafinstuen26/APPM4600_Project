import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm
import time

def driver():
    #THIS CODE IS FOR DIVERGENT NEWTON'S

    Nmax = 100
    x0 = np.array([0,0,0])
    tol = 1e-6
    
    [xstar, gval, ier,errors] = NewtonMethod(x0, tol, Nmax)
    print("The Newton method found the solution:", xstar)
    print("g evaluated at this point is:", gval)
    print("ier is:", ier)
    

def evalF(x):
    # F(X)
    F = np.zeros(3)
    F[0] = x[0]**2 - x[1]
    F[1] = x[0] * x[1] + x[1]**2
    F[2] = x[2]**3
    return F


def evalJ(x):
    # JACOBIAN
    J = np.array([
        [2*x[1],-1,0],
        [x[1],x[0]+2*x[1],0],
        [0,0,3*x[2]**2]
    ])
    return J

def evalg(x):
    # g(x) = ||F(x)||^2

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    # GRADIENT
    F = evalF(x)
    J = evalJ(x)
    return 2 * np.dot(J.T, F)

def eval_hessianf(x):
    # H(X)
    F = evalF(x)
    J = evalJ(x)
    
    n = len(F)
    m = len(x)
    second_derivatives = np.zeros((m, m))
    for i in range(n):
        Fi_hessian = evalH(x,i)  # Hessian of each F_i
        second_derivatives += 2 * F[i] * Fi_hessian
    
    # Full Hessian
    return 2 * np.dot(J.T, J) + second_derivatives

def evalH(x, i):
    # HESSIAN OF THE I-TH COMPONENT OF F(X)

    if i == 0:  # F_0(x) = x_0^2 - x_1
        return np.array([
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    elif i == 1:  # F_1(x) = x_0 x_1 + x_1^2
        return np.array([
            [0, 1, 0],
            [1, 2, 0],
            [0, 0, 0]
        ])
    elif i == 2:  # F_2(x) = x_2^3
        return np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 6 * x[2]]
        ])


def NewtonMethod(x, tol, Nmax):
    errors = []
    for its in range(Nmax):
        gradf = eval_gradg(x)
        H = eval_hessianf(x)
        print(H)

        try:
            delta = -np.linalg.solve(H, gradf)
        except np.linalg.LinAlgError:
            print("Singular Hessian matrix")
            ier = 1
            return [x, evalg(x), ier, errors]
        
        # Update solution
        x = x + delta
        
        # Evaluate the function norm to check convergence
        
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x, gval, ier,errors]
    
    print('Max iterations exceeded')
    ier = 1
    return [x, evalg(x), ier,errors]

if __name__ == '__main__':
    driver()
