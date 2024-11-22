import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import time

def driver():
    # THIS CODE IS FOR LAZY NEWTON'S METHOD

    Nmax = 100
    x0 = np.array([1.5, 0.5, 0])  # Initial guess
    tol = 1e-6
    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors] = LazyNewtonMethod(x0, tol, Nmax)
    end_time = time.perf_counter()
    print("The Lazy Newton method found the solution:", xstar)
    print("g evaluated at this point is:", gval)
    print("ier is:", ier)
    print("Time elapsed:", end_time - start_time)

    iterations = range(len(errors))
    convergence_order(errors[:-1], gval)
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, errors, color='dodgerblue', marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$e_k$', fontsize=14)
    plt.title("Lazy Newton Method Performance", fontsize=16)
    plt.show()

def evalF(x):
    # F(X)
    F = np.zeros(3)
    F[0] = x[0]**2 + x[1] + x[2] - 3
    F[1] = x[0] + x[1]**2 - x[2] - 2
    F[2] = x[0] - x[1] + x[2]**2 - 1
    return F

def evalJ(x):
    # JACOBIAN
    J = np.array([
        [2 * x[0], 1, 1],
        [1, 2 * x[1], -1],
        [1, -1, 2 * x[2]]
    ])
    return J

def evalg(x):
    # g(x) = ||F(x)||^2
    F = evalF(x)
    return np.dot(F, F)

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
        Fi_hessian = evalH(x, i)  # Hessian of each F_i
        second_derivatives += 2 * F[i] * Fi_hessian
    
    # Full Hessian
    return 2 * np.dot(J.T, J) + second_derivatives

def evalH(x, i):
    # HESSIAN OF THE I-TH COMPONENT OF F(X)
    if i == 0:  # F_0(x) = x_0^2 + x_1 + x_2 - 3
        return np.array([
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    elif i == 1:  # F_1(x) = x_0 + x_1^2 - x_2 - 2
        return np.array([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
    elif i == 2:  # F_2(x) = x_0 - x_1 + x_2^2 - 1
        return np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2]
        ])

def convergence_order(errors, gval):
    diff1 = np.abs(errors[1:])
    diff2 = np.abs(errors[:-1])
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
    print('The order equation is:')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))
    return [fit, diff1, diff2]

def LazyNewtonMethod(x, tol, Nmax):
    # SOLVING F(X) = 0 USING LAZY NEWTON'S METHOD
    errors = []
    H = eval_hessianf(x)  # Compute the Hessian initially
    prev_gval = None  # Track the previous gradient norm

    for its in range(Nmax):
        gradf = eval_gradg(x)

        # Check for convergence
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x, gval, ier, errors]

        # Decide whether to update the Hessian
        if prev_gval is not None and np.abs(gval - prev_gval) < 1e-3:
            H = eval_hessianf(x)  # Recompute the Hessian only when needed

        try:
            delta = -np.linalg.solve(H, gradf)
        except np.linalg.LinAlgError:
            print("Lazy Newton method: singular Hessian.")
            ier = 1
            return [x, evalg(x), ier, errors]

        # Update the solution
        x = x + delta
        prev_gval = gval  # Update the previous gradient norm

    print('Max iterations exceeded')
    ier = 1
    return [x, evalg(x), ier, errors]

if __name__ == '__main__':
    driver()