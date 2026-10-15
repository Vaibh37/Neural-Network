"""Reproducible dense-layer forward passes. No learning or weight updates."""
import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Apply ReLU elementwise without changing the array shape."""
    return np.maximum(values, 0)


def dense_forward(weights: np.ndarray, x: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pre-activation Z and activated A for column-oriented samples."""
    if weights.ndim != 2 or x.ndim != 2 or bias.ndim != 2:
        raise ValueError("Use two-dimensional arrays for W, X and b.")
    if weights.shape[1] != x.shape[0] or bias.shape != (weights.shape[0], 1):
        raise ValueError("Incompatible W, X or bias shapes.")
    z = weights @ x + bias
    return z, relu(z)


if __name__ == "__main__":
    W = np.array([[2, -1], [-1, 1]])
    x = np.array([[3], [4]])
    b = np.array([[1], [-2]])
    Z, A = dense_forward(W, x, b)
    assert Z.tolist() == [[3], [-1]]
    assert A.tolist() == [[3], [0]]
    print("Two neurons: Z =", Z.ravel(), "A =", A.ravel(), "shape =", A.shape)

    W3 = np.array([[2, 1], [-1, 3], [1, -2]])
    x3 = np.array([[2], [-1]])
    b3 = np.array([[1], [2], [-3]])
    Z3, A3 = dense_forward(W3, x3, b3)
    assert Z3.tolist() == [[4], [-3], [1]]
    assert A3.tolist() == [[4], [0], [1]]
    assert A3.shape == (3, 1)
    print("Three neurons: Z =", Z3.ravel(), "A =", A3.ravel(), "shape =", A3.shape)

    # Batch of two input columns: the same bias is broadcast across both.
    batch = np.array([[2, 3], [-1, 4]])
    Z_batch, A_batch = dense_forward(W3, batch, b3)
    assert A_batch.shape == (3, 2)
    print("Two-sample batch output shape =", A_batch.shape)
