import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def driver():
    # THIS CODE IS FOR BFGS
    Nmax = 100
    x0 = np.array([1.1, .93, .99])
    tol = 1e-12
    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors, evals] = BFGS(x0, tol, Nmax)
    end_time = time.perf_counter()
    print("xstar:", xstar)
    print("f(xstar):", eval_f(xstar))
    print("||", chr(8711), "f(xstar)||:", gval)
    print("ier:", ier)
    print("Time elapsed:", end_time - start_time)
    diff = np.array(evals)-np.array([0,0,0])
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
    plt.title("Semiology Error Plot of Newton's on $f(x)$", fontsize=16)
    plt.show()

def eval_f(x):
    return np.exp(x[0])-x[0] + np.exp(x[1])-x[1] + np.exp(x[2])-x[2]

def evalF(x):
    F = np.zeros(3)
    F[0] = np.exp(x[0]) - 1
    F[1] = np.exp(x[1]) - 1
    F[2] = np.exp(x[2]) - 1
    return F

def evalJ(x):
    J = np.array([
        [np.exp(x[0]), 0, 0],
        [0, np.exp(x[1]), 0],
        [0, 0, np.exp(x[2])]
    ])
    return J

def eval_norm(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
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
            [np.exp(x[0]), 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    elif i == 1:
        return np.array([
            [0, 0, 0],
            [0, np.exp(x[1]), 0],
            [0, 0, 0]
        ])
    elif i == 2:
        return np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, np.exp(x[2])]
        ])

def convergence_order(errors, gval):
    if len(errors) >3:
        errors = errors[-3:]
    diff1 = np.abs(errors[1:])
    diff2 = np.abs(errors[:-1])
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    print(r'asymptotic convergence order:', chr(945),'=', str(fit[0]))
    return [fit, diff1, diff2]

def BFGS(x, tol, Nmax):
    errors = []
    evals = []
    n = len(x)
    B = np.eye(n)  # Initial Hessian approximation as identity matrix

    def backtracking_line_search(f, gradf, x, delta, alpha=1.0, beta=0.8, c=1e-4):
        """Backtracking line search to find the optimal step size."""
        fx = f(x)
        gradfx = gradf(x)
        while f(x + alpha * delta) > fx + c * alpha * np.dot(gradfx, delta):
            alpha *= beta
        return alpha

    for its in range(Nmax):
        gradf = eval_gradg(x)  # Gradient at current x
        evals.append(np.array(gradf))

        # Solve for search direction
        try:
            delta = -np.linalg.solve(B, gradf)
        except np.linalg.LinAlgError:
            print("Singular approximation matrix B")
            ier = 1
            return [x, norm(gradf), ier, errors, evals]
        
        # Perform line search to determine step size
        alpha = backtracking_line_search(eval_f, eval_gradg, x, delta)
        
        # Update solution
        x_new = x + alpha * delta

        # Check for convergence
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x_new, gval, ier, errors, evals]
        
        # Update Hessian approximation using BFGS formula
        gradf_new = eval_gradg(x_new)
        s = x_new - x
        y = gradf_new - gradf
        Bs = np.dot(B, s)
        ys = np.dot(y, s)
        sBs = np.dot(s, Bs)
        
        # BFGS update rule
        B = B + np.outer(y, y) / ys - np.outer(Bs, Bs) / sBs

        # Move to next iteration
        x = x_new

    print('Max iterations exceeded')
    ier = 1
    return [x, gval, ier, errors, evals]



if __name__ == '__main__':
    driver()
