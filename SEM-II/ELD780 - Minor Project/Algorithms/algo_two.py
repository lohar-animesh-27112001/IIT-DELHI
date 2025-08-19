import numpy as np

def smoothed_weiszfeld(points, alphas=None, nu=1e-6, max_iter=100, tol=1e-6, init=None):
    points = np.asarray(points)
    n_points, dim = points.shape
    if alphas is None:
        alphas = np.ones(n_points)/n_points
    else:
        alphas = np.asarray(alphas)
        alphas /= alphas.sum()
    if init is None:
        estimate = np.average(points, axis=0, weights=alphas)
    else:
        estimate = np.asarray(init).copy()
    for _ in range(max_iter):
        residuals = points - estimate
        distances = np.linalg.norm(residuals, axis=1)
        weights = alphas / np.maximum(nu, distances)
        weights /= weights.sum()
        new_estimate = np.sum(weights[:, None] * points, axis=0)
        update_norm = np.linalg.norm(new_estimate - estimate)
        estimate = new_estimate
        if update_norm < tol:
            break
    return estimate

if __name__ == "__main__":
    points = [
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0]
    ]
    alphas = [1/3, 1/3, 1/3]
    median = smoothed_weiszfeld(points, alphas, nu=0.1)
    print(f"Geometric median (Test 1): {median}")
    points = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0]
    ]
    alphas = [0.3, 0.3, 0.3, 0.1]
    median = smoothed_weiszfeld(points, alphas)
    print(f"Geometric median (Test 2): {median}")
    points = [[1.0, 2.0]] * 5
    median = smoothed_weiszfeld(points)
    print(f"Geometric median (Test 3): {median}")