import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sympy as sym

def driver():
    # THIS CODE IS FOR NEWTON'S ON GAUSS FUNCTION

    Nmax = 100
    x0 = np.array([0.85, -1.,0.01])
    tol = 1e-12

    plot_3d(cost,False)
    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors, evals] = NewtonMethod(x0, tol, Nmax)
    end_time = time.perf_counter()
    print("xstar:", xstar)
    print("f(xstar):", cost(xstar))
    print("||", chr(8711), "F(xstar)||:", gval)
    print("ier:", ier)
    print("Time elapsed:", end_time - start_time)

    iterations = range(len(errors))
    convergence_order(errors[:-1], gval)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(iterations, np.abs(errors), color='dodgerblue', marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$e_k$', fontsize=14)
    plt.title("Semiology Error Plot of Newton's on $F(x)$", fontsize=16)
    plt.show()

    plot_3d_2(cost,True,evals)
    plot_contour(cost,evals)

    # Converging to flat
    #x0 = np.array([0.85, -1.,0.4])
    x0 = np.array([1.6, -1.8,.8])
    [xstar, gval, ier, errors, evals] = NewtonMethod(x0, tol, Nmax)
    print("xstar:", xstar)
    print("f(xstar):", cost(xstar))
    print("||", chr(8711), "F(xstar)||:", gval)
    print("ier:", ier)

    iterations = range(len(errors))
    convergence_order(errors[:-1], gval)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(iterations, np.abs(errors), color='dodgerblue', marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$e_k$', fontsize=14)
    plt.title("Semiology Error Plot of Newton's on $F(x)$", fontsize=16)
    plt.show()

    plot_3d_3(cost,True,evals)


    #plot_domain(cost)
    #plot_domain_paths(cost)

def generate():
    x,y,z = sym.symbols("x y z")
    expr = - sym.exp(-(x-1)**2-(y+2)**2-z**2) - sym.exp(-(x+1)**2- (y+1)**2-z**2)

    F0 = expr.diff(x)
    F1 = expr.diff(y)
    F2 = expr.diff(z)

    J00 = sym.lambdify((x,y,z),F0.diff(x))
    J01 = sym.lambdify((x,y,z),F0.diff(y))
    J02 = sym.lambdify((x,y,z),F0.diff(z))
    J10 = sym.lambdify((x,y,z),F1.diff(x))
    J11 = sym.lambdify((x,y,z),F1.diff(y))
    J12 = sym.lambdify((x,y,z),F1.diff(z))
    J20 = sym.lambdify((x,y,z),F2.diff(x))
    J21 = sym.lambdify((x,y,z),F2.diff(y))
    J22 = sym.lambdify((x,y,z),F2.diff(z))


    H0_00 = sym.lambdify((x,y,z),F0.diff(x).diff(x))
    H0_01 = sym.lambdify((x,y,z),F0.diff(x).diff(y))
    H0_02 = sym.lambdify((x,y,z),F0.diff(x).diff(z))
    H0_10 = sym.lambdify((x,y,z),F0.diff(y).diff(x))
    H0_11 = sym.lambdify((x,y,z),F0.diff(y).diff(y))
    H0_12 = sym.lambdify((x,y,z),F0.diff(y).diff(z))
    H0_20 = sym.lambdify((x,y,z),F0.diff(z).diff(x))
    H0_21 = sym.lambdify((x,y,z),F0.diff(z).diff(y))
    H0_22 = sym.lambdify((x,y,z),F0.diff(z).diff(z))

    H1_00 = sym.lambdify((x,y,z),F1.diff(x).diff(x))
    H1_01 = sym.lambdify((x,y,z),F1.diff(x).diff(y))
    H1_02 = sym.lambdify((x,y,z),F1.diff(x).diff(z))
    H1_10 = sym.lambdify((x,y,z),F1.diff(y).diff(x))
    H1_11 = sym.lambdify((x,y,z),F1.diff(y).diff(y))
    H1_12 = sym.lambdify((x,y,z),F1.diff(y).diff(z))
    H1_20 = sym.lambdify((x,y,z),F1.diff(z).diff(x))
    H1_21 = sym.lambdify((x,y,z),F1.diff(z).diff(y))
    H1_22 = sym.lambdify((x,y,z),F1.diff(z).diff(z))

    H2_00 = sym.lambdify((x,y,z),F2.diff(x).diff(x))
    H2_01 = sym.lambdify((x,y,z),F2.diff(x).diff(y))
    H2_02 = sym.lambdify((x,y,z),F2.diff(x).diff(z))
    H2_10 = sym.lambdify((x,y,z),F2.diff(y).diff(x))
    H2_11 = sym.lambdify((x,y,z),F2.diff(y).diff(y))
    H2_12 = sym.lambdify((x,y,z),F2.diff(y).diff(z))
    H2_20 = sym.lambdify((x,y,z),F2.diff(z).diff(x))
    H2_21 = sym.lambdify((x,y,z),F2.diff(z).diff(y))
    H2_22 = sym.lambdify((x,y,z),F2.diff(z).diff(z))


    num_expr = sym.lambdify((x,y,z),expr)
    num_F0 = sym.lambdify((x,y,z),F0)
    num_F1 = sym.lambdify((x,y,z),F1)
    num_F2 = sym.lambdify((x,y,z),F2)
    

    def cost(arr):
        return num_expr(*arr)
    
    def evalF(arr):
        return np.array([
            num_F0(*arr),
            num_F1(*arr),
            num_F2(*arr),
        ])
    
    def evalJ(arr):
        return np.array([
            [J00(*arr),J01(*arr),J02(*arr)],
            [J10(*arr),J11(*arr),J12(*arr)],
            [J20(*arr),J21(*arr),J22(*arr)],
        ])
    
    def evalH(arr,i):
        if i==0:
            return np.array([
                [H0_00(*arr),H0_01(*arr),H0_02(*arr)],
                [H0_10(*arr),H0_11(*arr),H0_12(*arr)],
                [H0_20(*arr),H0_21(*arr),H0_22(*arr)],
            ])
        elif i==1:
            return np.array([
                [H1_00(*arr),H1_01(*arr),H1_02(*arr)],
                [H1_10(*arr),H1_11(*arr),H1_12(*arr)],
                [H1_20(*arr),H1_21(*arr),H1_22(*arr)],
            ])
        elif i ==2:
            return np.array([
                [H2_00(*arr),H2_01(*arr),H2_02(*arr)],
                [H2_10(*arr),H2_11(*arr),H2_12(*arr)],
                [H2_20(*arr),H2_21(*arr),H2_22(*arr)],
            ])

    return cost,evalF,evalJ,evalH

cost,evalF,evalJ,evalH = generate()


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


def convergence_order(errors, gval):
    if len(errors)>3:
        errors = errors[-3:]
    diff1 = np.abs(errors[1:])
    diff2 = np.abs(errors[:-1])
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()), 1)
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
            #print("Singular Hessian matrix")
            ier = 1
            return [x, norm(gradf), ier, errors, evals]
        
        x += delta
        gval = norm(gradf)
        errors.append(gval)
        if abs(gval) < tol:
            ier = 0
            return [x, gval, ier, errors, evals]
    
    #print('Max iterations exceeded')
    ier = 1
    return [x, gval, ier, errors, evals]

def plot_3d(func, ran, points = []):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid for the surface plot
    x1 = np.linspace(-4, 4, 200)
    x2 = np.linspace(-5, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0  # Fix x3 at 0 for the surface plot
    Z = func([X1, X2, X3])

    # Plot the surface
    ax.plot_surface(X1, X2, Z, cmap='Spectral', alpha=0.8)

    # Plot the Newton's method points
    if ran:
         # Convert points to numpy array for processing
        points[0][2] = 0
        points = np.array(points)

        # Extract x, y, z values for the points
        xs = points[:, 0]
        ys = points[:, 1]
        zs = [func(point) for point in points]  # Evaluate func for the z-coordinates

        ax.scatter(xs, ys, zs, color='midnightblue', s=50, label="Newton's Method Points")
        ax.plot(xs, ys, zs, color='dodgerblue', linestyle='--', label="Optimization Path")

    # Set plot labels and title
        plt.suptitle("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$ with Newton's Method Iterations", fontsize=16, y= .9)
    else: 
        plt.suptitle("3D Plot of $f(x) = e^{-(x_1-1)^2-(x_2+2)^2-x_3^2} - e^{-(x_1+1)^2-(x_2+1)^2-x_3^2}$",fontsize=16, y=.9)
        #ax.set_title("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$", fontsize=14)
        
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    plt.title("$x_3 = 0$", fontsize = 13, y=1)

    # Show the plot
    plt.show()

def plot_3d_2(func, ran, points = []):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid for the surface plot
    x1 = np.linspace(0, 2, 200)
    x2 = np.linspace(-2.75, -.75, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0  # Fix x3 at 0 for the surface plot
    Z = func([X1, X2, X3])

    # Plot the surface
    ax.plot_surface(X1, X2, Z, cmap='Spectral', alpha=0.8)

    # Plot the Newton's method points
    if ran:
         # Convert points to numpy array for processing
        points[0][2] = 0
        points = np.array(points)

        # Extract x, y, z values for the points
        xs = points[:, 0]
        ys = points[:, 1]
        zs = [func(point) for point in points]  # Evaluate func for the z-coordinates

        # Plot all points except the last one in midnight blue
        ax.scatter(xs[:-1], ys[:-1], zs[:-1], color='midnightblue', s=50, label="Newton's Method Points")

        # Highlight the last point in dodger blue
        ax.scatter(xs[-1], ys[-1], zs[-1], color='dodgerblue', s=50, label="Last Point")

        ax.plot(xs, ys, zs, color='dodgerblue', linestyle='--', label="Optimization Path")

    # Set plot labels and title
        plt.suptitle("3D Plot of $f(x)$ with Newton's Method Iterations", fontsize=16, y= .8)
    else: 
        plt.suptitle("3D Plot of $f(x) = e^{-(x_1-1)^2-(x_2+2)^2-x_3^2} - e^{-(x_1+1)^2-(x_2+1)^2-x_3^2}$",fontsize=16, y=.9)
        #ax.set_title("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$", fontsize=14)
        
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    plt.title("$x_3 = 0$", fontsize = 13, y=.86)

    # Show the plot
    plt.show()

def plot_3d_3(func, ran, points = []):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid for the surface plot
    x1 = np.linspace(-5, 5, 200)
    x2 = np.linspace(-5, 5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0  # Fix x3 at 0 for the surface plot
    Z = func([X1, X2, X3])

    # Plot the surface
    ax.plot_surface(X1, X2, Z, cmap='Spectral', alpha=0.8)

    # Plot the Newton's method points
    if ran:
         # Convert points to numpy array for processing
        points[0][2] = 0
        points = np.array(points)

        # Extract x, y, z values for the points
        xs = points[:, 0]
        ys = points[:, 1]
        zs = [func(point) for point in points]  # Evaluate func for the z-coordinates

        # Plot all points except the last one in midnight blue
        ax.scatter(xs[:-1], ys[:-1], zs[:-1], color='midnightblue', s=50, label="Newton's Method Points")

        # Highlight the last point in dodger blue
        ax.scatter(xs[-1], ys[-1], zs[-1], color='dodgerblue', s=50, label="Last Point")

        ax.plot(xs, ys, zs, color='dodgerblue', linestyle='--', label="Optimization Path")

    # Set plot labels and title
        plt.suptitle("3D Plot of $f(x)$ with Newton's Method Iterations", fontsize=16, y= .9)
    else: 
        plt.suptitle("3D Plot of $f(x) = e^{-(x_1-1)^2-(x_2+2)^2-x_3^2} - e^{-(x_1+1)^2-(x_2+1)^2-x_3^2}$",fontsize=16, y=.9)
        #ax.set_title("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$", fontsize=14)
        
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    plt.title("$x_3 = 0$", fontsize = 13, y=1)

    # Show the plot
    plt.show()

def plot_contour(func, points):
    fig, ax = plt.subplots(figsize=(8, 6))

    x1 = np.linspace(-1.5, 2.5, 200)
    x2 = np.linspace(-3.5, .5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0
    Z = func([X1, X2, X3])

    contour = ax.contourf(X1, X2, Z, levels=100, cmap='Spectral')
    plt.colorbar(contour, ax=ax, label="$f(x)$")
    points = np.array(points)
    for i in range(len(points) - 1):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], 
                color='dodgerblue', linestyle='--')
        ax.plot(points[i][0], points[i][1], 'o', color='midnightblue')

    ax.set_title("Contour Map of $f(x)$ with Newton's Method Iterations", fontsize=14)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.show()

def plot_domain(func):
    x1_min,x1_max = -2,2
    num_x1 = 401
    x2_min,x2_max = -3.5,0.5
    num_x2 = 401
    X1, X2 = np.meshgrid(
        np.linspace(x1_min,x1_max,num_x1),
        np.linspace(x2_min,x2_max,num_x2),
    )

    converged = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        xstar, gval, ier, errors, evals = NewtonMethod(np.array([x1,x2,0]), 1e-12, 1000)
        if ier == 0 and cost(xstar)<-1:
            if norm(xstar-np.array([1,-2,0]))<1e-1:
                converged.append(2.9)
            elif norm(xstar-np.array([-1,-1,0]))<1e-1:
                converged.append(9)
            else:
                converged.append(-50)
       # elif ier == 0:
            #converged.append(1)
        else:
            converged.append(0)

    converged = np.array(converged).reshape(X1.shape)
    z = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        z.append(cost(np.array([x1,x2,0])))
    z = np.array(z).reshape(X1.shape)
    plt.pcolormesh(X1,X2,converged, cmap = 'jet') #RdYlBu
    contour = plt.contour(X1,X2,z, cmap = 'PuRd')
    plt.clabel(contour,contour.levels)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Contour Map of the domain of convergence for $f(x)$")
    

    plt.show()

def plot_domain_paths(func):
    x1_min,x1_max = -2,2
    num_x1 = 401
    x2_min,x2_max = -3.5,0.5
    num_x2 = 401
    X1, X2 = np.meshgrid(
        np.linspace(x1_min,x1_max,num_x1),
        np.linspace(x2_min,x2_max,num_x2),
    )

    converged = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        xstar, gval, ier, errors, evals = NewtonMethod(np.array([x1,x2,0]), 1e-12, 1000)
        if ier == 0 and cost(xstar)<-1:
            if norm(xstar-np.array([1,-2,0]))<1e-1:
                converged.append(2.9)
            elif norm(xstar-np.array([-1,-1,0]))<1e-1:
                converged.append(9)
            else:
                converged.append(-50)
       # elif ier == 0:
            #converged.append(1)
        else:
            converged.append(0)

    converged = np.array(converged).reshape(X1.shape)
    z = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        z.append(cost(np.array([x1,x2,0])))
    z = np.array(z).reshape(X1.shape)
    plt.pcolormesh(X1,X2,converged, cmap = 'jet') #RdYlBu
    contour = plt.contour(X1,X2,z, cmap = 'PuRd')
    plt.clabel(contour,contour.levels)

    xs = np.array([np.array([-.25,-1.3,0]),np.array([-1.5,-1.858,0]),np.array([-1.2886,-.897,0]),np.array([-0.159,-1.222,0]),np.array([-1.8989,-.473,0])])
    for i in tqdm(range(len(xs))):
        xstar, gval, ier, errors, evals = NewtonMethod(xs[i], 1e-12, 1000)
        plt.plot([x[0] for x in evals], [x[1] for x in evals], '.-')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Contour Map of the domain of convergence for $f(x)$ with convergence paths")
    plt.show()
    

if __name__ == '__main__':
    driver()
