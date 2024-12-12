import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import norm
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def driver():
    # THIS CODE IS FOR NEWTON'S ON STRICTLY CONVEX FUNCTION R3

    Nmax = 100
    x0 = np.array([1.5, 1.6, .8])
    tol = 1e-12

    #plot_domain(eval_f)

    plot_3d(eval_f,False)
    #plot_3d_2(eval_f,False)
    
    start_time = time.perf_counter()
    [xstar, gval, ier, errors, evals] = NewtonMethod(x0, tol, Nmax)
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
    plt.title("Semiology Error Plot of Newton's on $F(x)$", fontsize=16)
    plt.show()

    # 3D Plot of eval_ross and points from Newton's method
    plot_3d(eval_f, True,evals)
    # Contour Plot of eval_ross and points from Newton's method
    plot_contour(eval_f, evals)

    plot_planes_with_point(xstar)
    plot_planes_and_point(xstar)

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
    
    #print('Max iterations exceeded')
    ier = 1
    return [x, gval, ier, errors, evals]

def plot_3d(func, ran, points=[]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid for the surface plot
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0  # Fix x3 at 1 for the surface plot
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

        # Plot the path in dodger blue
        ax.plot(xs, ys, zs, color='dodgerblue', linestyle='--', label="Optimization Path")

    # Set plot labels and title
        plt.suptitle("3D Plot of $f(x)$ with Newton's Method Iterations", fontsize=16, y=0.9)
    else:
        plt.suptitle("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$", fontsize=16, y=0.9)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    plt.title("$x_3 = 0$", fontsize=13, y=1)

    # Show the plot
    plt.show()

def plot_3d_2(func, ran, points=[]):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a grid for the surface plot
    x1 = np.linspace(-50, 30, 100)
    x2 = np.linspace(-50, 30, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 0  # Fix x3 at 1 for the surface plot
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

        # Plot the path in dodger blue
        ax.plot(xs, ys, zs, color='dodgerblue', linestyle='--', label="Optimization Path")

    # Set plot labels and title
        plt.suptitle("3D Plot of $f(x)$ with Newton's Method Iterations", fontsize=16, y=0.9)
    else:
        plt.suptitle("3D Plot of $f(x) = e^{x_1}-x_1 + e^{x_2}-x_2 + e^{x_3}-x_3$", fontsize=16, y=0.)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")
    plt.title("$x_3 = 0$", fontsize=13, y=1)

    # Show the plot
    plt.show()


def plot_contour(func, points):
    fig, ax = plt.subplots(figsize=(8, 6))

    x1 = np.linspace(-2, 2, 200)
    x2 = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = 1
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

def plot_planes_with_point(point):
    """
    Plots 3D planes (F1, F2, F3) and a point in separate figures.
    
    Parameters:
    - point: List or tuple with the coordinates of the point to be plotted [x, y, z].
    - x_range: Tuple defining the range of x-axis values (default: (-3, 3)).
    - y_range: Tuple defining the range of y-axis values (default: (-3, 3)).
    - z_range: Tuple defining the range of z-axis values (default: (-3, 3)).
    - resolution: Number of points for the grid along each axis (default: 100).
    """
    x_range=(-3, 3)
    y_range=(-3, 3)
    z_range=(-3, 3)
    
    # Define the functions representing the planes
    def F1_plane(x, y):
        return np.exp(x)-1 # Solving F1(x, y, z) = 0 for z

    def F2_plane(x, y):
        return np.exp(y)-1  # Solving F2(x, y, z) = 0 for z

    def F3_plane(y, z):
        return np.exp(z)-1  # Solving F3(x, y, z) = 0 for x

    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = F1_plane(X, Y)
    Z2 = F2_plane(X, Y)

    # Create grid for F3 with x as the dependent variable
    z = np.linspace(*z_range, 100)
    Y_F3, Z_F3 = np.meshgrid(y, z)
    X_F3 = F3_plane(Y_F3, Z_F3)

    # Create the figure with three subplots
    fig = plt.figure(figsize=(18, 6))

    # Plot F1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z1, alpha=0.5, cmap="Spectral", edgecolor='none')
    ax1.scatter(point[0], point[1], point[2], color="midnightblue", s=100, label="$x^*$", marker='o', edgecolor='midnightblue', linewidth=1.5)
    ax1.set_title(chr(8711) + "$f_1(x)$", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X-axis", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Y-axis", fontsize=12, fontweight='bold')
    ax1.set_zlabel("Z-axis", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Plot F2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z2, alpha=0.5, cmap="Spectral", edgecolor='none')
    ax2.scatter(point[0], point[1], point[2], color="midnightblue", s=100, label="$x^*$", marker='o', edgecolor='midnightblue', linewidth=1.5)
    ax2.set_title(chr(8711) + "$f_2(x)$", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X-axis", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Y-axis", fontsize=12, fontweight='bold')
    ax2.set_zlabel("Z-axis", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Plot F3
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_F3, Y_F3, Z_F3, alpha=0.5, cmap="Spectral", edgecolor='none')
    ax3.scatter(point[0], point[1], point[2], color="midnightblue", s=100, label="$x^*$", marker='o', edgecolor='midnightblue', linewidth=1.5)
    ax3.set_title(chr(8711) + "$f_3(x)$", fontsize=14, fontweight='bold')
    ax3.set_xlabel("X-axis", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Y-axis", fontsize=12, fontweight='bold')
    ax3.set_zlabel("Z-axis", fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()


def plot_planes_and_point(point):
    """
    Plots 3D planes (F1, F2, F3) and a point on the same 3D plot.
    
    Parameters:
    - point: List or tuple with the coordinates of the point to be plotted [x, y, z].
    - x_range: Tuple specifying the range of x-axis values (default: (-2, 2)).
    - y_range: Tuple specifying the range of y-axis values (default: (-2, 2)).
    - resolution: Number of points for the grid along each axis (default: 100).
    """
    # Define the functions representing the planes
    x_range = (-3, 3)
    y_range = (-3, 3)
    z_range = (-3, 3)

    # Define the functions representing the planes
    def F1_plane(x, y):
        return np.exp(x) - 1  # Solving F1(x, y, z) = 0 for z

    def F2_plane(x, y):
        return np.exp(y) - 1  # Solving F2(x, y, z) = 0 for z

    def F3_plane(y, z):
        return np.exp(z) - 1  # Solving F3(x, y, z) = 0 for x

    # Create a grid for plotting
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)

    # Compute Z values for each plane
    Z1 = F1_plane(X, Y)
    Z2 = F2_plane(X, Y)
    
    z = np.linspace(*z_range, 100)
    Y_F3, Z_F3 = np.meshgrid(y, z)
    X_F3 = F3_plane(Y_F3, Z_F3)

    # Plot the planes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Add the planes with vibrant colormaps
    ax.plot_surface(X, Y, Z1, alpha=0.4, cmap="viridis", rstride=5, cstride=5, edgecolor='none')
    ax.plot_surface(X, Y, Z2, alpha=0.4, cmap="plasma", rstride=5, cstride=5, edgecolor='none')
    ax.plot_surface(X_F3, Y_F3, Z_F3, alpha=0.4, cmap="cividis", rstride=5, cstride=5, edgecolor='none')

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color="midnightblue", s=200, label=f"$x^*$ {point}", marker='o', edgecolor='midnightblue', linewidth=1.5)

    # Add labels and legend
    ax.set_title(chr(8711) + "$f(x)$" + " with $x^*$", fontsize=14, y = .9)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_zlabel("Z-axis", fontsize=12)

    # Add legend and enhance tick styles
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Show the plot
    plt.show()

def plot_domain(func):
    x1_min,x1_max = -30,15
    num_x1 = 1001
    x2_min,x2_max = -30,15
    num_x2 = 1001
    X1, X2 = np.meshgrid(
        np.linspace(x1_min,x1_max,num_x1),
        np.linspace(x2_min,x2_max,num_x2),
    )

    converged = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        xstar, gval, ier, errors, evals = NewtonMethod(np.array([x1,x2,0]), 1e-12, 20)
        if ier == 0:
            converged.append(9)
        else:
            converged.append(0)

    converged = np.array(converged).reshape(X1.shape)
    z = []
    for x1,x2 in tqdm(list(zip(X1.ravel(),X2.ravel()))):
        z.append(eval_f(np.array([x1,x2,0])))
    z = np.array(z).reshape(X1.shape)
    plt.pcolormesh(X1,X2,converged, cmap = 'jet') #RdYlBu
    contour = plt.contour(X1,X2,z, cmap = 'PuRd')
    plt.clabel(contour,contour.levels)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Contour Map of the domain of convergence for $f(x)$")
    


if __name__ == '__main__':
    driver()
