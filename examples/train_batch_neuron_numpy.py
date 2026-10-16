"""Train one linear neuron on four examples using full-batch gradient descent.

Checkpoint: 16 October 2026. This is a linear regression learning exercise,
not a hidden-layer network or the planned XOR classifier.

Run: python examples/train_batch_neuron_numpy.py
"""
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 10.0, 15.0, 20.0])

w = 3.0
b = 1.0
learning_rate = 0.05

initial_pred = w * x + b
initial_loss = np.mean((y - initial_pred) ** 2)
assert np.isclose(initial_loss, 21.0)
print(f"Initial MSE: {initial_loss:.6f}")

for epoch in range(5):
    # Predict all four targets using the SAME w and b.
    pred_y = w * x + b

    # MSE gradient: elementwise per-sample contributions, then mean.
    dw = 2 * np.mean((pred_y - y) * x)
    db = 2 * np.mean(pred_y - y)

    # One weight update and one bias update per full batch.
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Report POST-update loss (fresh predictions).
    pred_y = w * x + b
    loss = np.mean((y - pred_y) ** 2)
    print(f"Epoch {epoch + 1}: MSE={loss:.6f}, w={w:.6f}, b={b:.6f}")

assert pred_y.shape == (4,)
assert np.isclose(loss, 0.325768188171045)
assert loss < initial_loss
print("Checks passed.")
