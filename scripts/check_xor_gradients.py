"""Check the XOR network's backpropagation formulas with finite differences.

This uses a fixed 2 -> 3 -> 1 network with nonzero ReLU pre-activations
so the derivative is defined at every test point. No training is needed.
"""

import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])

# Hand-picked parameters keep the ReLU inputs away from zero.
W_h = np.array([[0.3, -0.4, 0.8], [-0.7, 0.6, 0.5]])
b_h = np.array([0.2, -0.3, 0.4])
W_o = np.array([[0.4], [-0.2], [0.7]])
b_o = np.array([-0.15])
parameters = {"W_h": W_h, "b_h": b_h, "W_o": W_o, "b_o": b_o}


def forward_and_gradients():
    """Return the MSE loss and the four analytic backpropagation gradients."""
    z_h = X @ W_h + b_h
    h = np.maximum(0.0, z_h)
    a_o = 1.0 / (1.0 + np.exp(-(h @ W_o + b_o)))
    loss = np.mean((a_o - y) ** 2)

    dL_dz_o = (2.0 / len(X)) * (a_o - y) * a_o * (1.0 - a_o)
    dW_o = h.T @ dL_dz_o
    db_o = np.sum(dL_dz_o, axis=0)
    dL_dz_h = (dL_dz_o @ W_o.T) * (z_h > 0.0)
    dW_h = X.T @ dL_dz_h
    db_h = np.sum(dL_dz_h, axis=0)
    return float(loss), {"W_h": dW_h, "b_h": db_h, "W_o": dW_o, "b_o": db_o}


def numerical_gradient(parameter, index, epsilon=1e-6):
    """Estimate d(loss)/d(parameter[index]) by central differences."""
    original = parameter[index]
    try:
        parameter[index] = original + epsilon
        plus, _ = forward_and_gradients()
        parameter[index] = original - epsilon
        minus, _ = forward_and_gradients()
    finally:
        parameter[index] = original
    return (plus - minus) / (2.0 * epsilon)


def main():
    _, gradients = forward_and_gradients()
    errors = []
    for name, parameter in parameters.items():
        analytic = gradients[name]
        for index in np.ndindex(parameter.shape):
            numeric = numerical_gradient(parameter, index)
            errors.append(abs(numeric - analytic[index]))
            np.testing.assert_allclose(
                numeric, analytic[index], rtol=1e-5, atol=1e-7,
                err_msg=f"Gradient mismatch for {name}{index}",
            )
    print(f"Checked {sum(p.size for p in parameters.values())} XOR gradients")
    print(f"Maximum absolute error: {max(errors):.2e}")


if __name__ == "__main__":
    main()
