import numpy as np


# 1. Mean Squared Error 


def compute_mse(y, tx, w):
    """Compute the Mean Squared Error."""
    e = y - tx @ w
    return (e @ e) / (2 * len(y))


# 2. Linear Regression using Gradient Descent


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = -(tx.T @ (y - tx @ w)) / len(y)
        w -= gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


# 3. Linear Regression using Stochastic Gradient Descent


def batch_iter(y, tx, batch_size=1, num_batches=1, seed=1):
    """Yield mini-batches of size 1 (SGD)."""
    np.random.seed(seed)
    N = len(y)
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        idx = indices[i:i + batch_size]
        yield y[idx], tx[idx]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1)."""
    w = initial_w.copy()
    for _ in range(max_iters):
        for y_b, tx_b in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = -(tx_b.T @ (y_b - tx_b @ w)) / len(y_b)
            w -= gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


# 4. Least Squares Regression (Normal Equation)


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


# 5. Ridge Regression


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, D = tx.shape
    I = np.eye(D)
    # bias term not regularized
    I[0, 0] = 0
    A = tx.T @ tx + 2 * N * lambda_ * I
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss


# 6. Logistic Regression (Gradient Descent)


def sigmoid(t):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    
    """Negative log-likelihood loss (without regularization)."""
    y_ = (y+1)/2
    pred = sigmoid(tx @ w)
    return -np.mean(y_ * np.log(pred + 1e-15) + (1 - y_) * np.log(1 - pred + 1e-15))

def compute_logistic_grad(y, tx, w):
    """Gradient of logistic loss."""
    pred = sigmoid(tx @ w)
    return tx.T @ ((pred - y)/2) / len(y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = compute_logistic_grad(y, tx, w)
        w -= gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


# 7. Regularized Logistic Regression


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        grad = compute_logistic_grad(y, tx, w) + 2 * lambda_ * w
        grad[0] -= 2 * lambda_ * w[0]  # don't regularize bias
        w -= gamma * grad
    loss = compute_logistic_loss(y, tx, w)  # exclude penalty term
    return w, loss
