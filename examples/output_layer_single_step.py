"""One 2→3→1 forward pass and one output-layer gradient-descent step.

This example intentionally holds all hidden-layer weights and biases fixed.
It demonstrates only output-layer gradients for one XOR training example.
"""

import numpy as np

# One XOR example: 1 XOR 0 = 1.
x = np.array([1.0, 0.0])
target = 1.0

# The weights below are chosen manually for an easy worked example.
W_hidden = np.array([
    [1.0, 1.0],
    [-1.0, 1.0],
    [2.0, -1.0],
])
b_hidden = np.array([0.0, 0.0, 0.0])

v = np.array([0.2, 0.4, 0.3])
b_out = 0.0
learning_rate = 0.05

# 1) FORWARD PROPAGATION: hidden layer, then linear output and squared loss.
z_hidden = W_hidden @ x + b_hidden
h = np.maximum(0.0, z_hidden)
prediction = float(v @ h + b_out)
loss = (prediction - target) ** 2

# 2) BACKPROPAGATION: calculate all output-layer gradients.
dL_dprediction = 2 * (prediction - target)
dL_dv = dL_dprediction * h               # ∂ŷ/∂vᵢ = hᵢ
dL_db_out = dL_dprediction * 1.0        # ∂ŷ/∂b_out = 1

# 3) GRADIENT DESCENT: update parameters using the gradients.
v = v - learning_rate * dL_dv
b_out = b_out - learning_rate * dL_db_out

# 4) FORWARD PASS AGAIN: same hidden activations, updated output parameters.
updated_prediction = float(v @ h + b_out)
updated_loss = (updated_prediction - target) ** 2

assert np.allclose(z_hidden, [1.0, -1.0, 2.0])
assert np.allclose(h, [1.0, 0.0, 2.0])
assert np.isclose(prediction, 0.8)
assert np.isclose(loss, 0.04)
assert np.isclose(dL_dprediction, -0.4)
assert np.allclose(dL_dv, [-0.4, 0.0, -0.8])
assert np.isclose(dL_db_out, -0.4)
assert np.allclose(v, [0.22, 0.40, 0.34])
assert np.isclose(b_out, 0.02)
assert np.isclose(updated_prediction, 0.92)
assert np.isclose(updated_loss, 0.0064)

print(f"Hidden weighted sums: {z_hidden}")
print(f"Hidden activations: {h}")
print(f"Before: prediction={prediction:.2f}, loss={loss:.4f}")
print(f"Output weight gradients: {dL_dv}")
print(f"Output bias gradient: {dL_db_out:.2f}")
print(f"After: prediction={updated_prediction:.2f}, loss={updated_loss:.4f}")
