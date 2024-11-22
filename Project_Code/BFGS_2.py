import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
import time

def driver():
    Nmax = 100
    x0 = np.array([1.5, 0.5, 0])
    tol = 1e-6
    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors, evals] = BFGSMethod(x0, tol, Nmax)
    end_time = time.perf_counter()
    print("The BFGS method found the solution:", xstar)
    print("g evaluated at this point is:", gval)
    print("ier is:", ier)
    print("Time elapsed:", end_time - start_time)

    iterations = range(len(errors))
    convergence_order(errors[:-1], gval)
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, errors, color='dodgerblue', marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$e_k$', fontsize=14)
    plt.title("BFGS Method Performance", fontsize=16)
    plt.show()

def evalF(x):
    F = np.zeros(3)
    F[0] = x[0]**2 + x[1] + x[2] - 3
    F[1] = x[0] + x[1]**2 - x[2] - 2
    F[2] = x[0] - x[1] + x[2]**2 - 1
    return F

def evalJ(x):
    J = np.array([
        [2*x[0], 1, 1],
        [1, 2*x[1], -1],
        [1, -1, 2*x[2]]
    ])
    return J

def evalg(x):
    F = evalF(x)
    return np.dot(F, F)

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    return 2 * np.dot(J.T, F)

def convergence_order(x, xstar):
    diff1 = np.abs(x[1::] - xstar)
    diff2 = np.abs(x[0:-1] - xstar)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    print('The order equation is:')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda =', str(np.exp(fit[1])))
    print('alpha =', str(fit[0]))
    return [fit, diff1, diff2]

def BFGSMethod(x, tol, Nmax):
    errors = []
    evals = []
    n = len(x)
    B = np.eye(n)  # Initial Hessian approximation as identity matrix

    for its in range(Nmax):
        gradf = eval_gradg(x)
        evals.append(np.array(gradf))

        # Solve for search direction
        try:
            delta = -np.linalg.solve(B, gradf)
        except np.linalg.LinAlgError:
            print("Singular approximation matrix B")
            ier = 1
            return [x, evalg(x), ier, errors, evals]
        
        # Update solution
        x_new = x + delta

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
        
        # BFGS update rule
        B = B + np.outer(y, y) / np.dot(y, s) - np.outer(Bs, Bs) / np.dot(s, Bs)

        # Move to next iteration
        x = x_new

    print('Max iterations exceeded')
    ier = 1
    return [x, evalg(x), ier, errors, evals]

if __name__ == '__main__':
    driver()