import numpy as np

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

# Random initialization
rng = np.random.default_rng(42)

# Hidden layer parameters
W_h = rng.normal(0, 0.5, size=(2, 3))
b_h = np.zeros(3)

# Output layer parameters
W_o = rng.normal(0, 0.5, size=(3, 1))
b_o = np.zeros(1)

# Learning rate
A = 0.05
epochs = 5000

for epoch in range(epochs):

    z_h = X @ W_h + b_h
    h = np.maximum(0, z_h)

    z_o = h @ W_o + b_o
    a_o = 1 / (1 + np.exp(-z_o))

    error = (a_o - y)
    squared_error = error ** 2
    loss = np.mean(squared_error)

    dl_da_o = 2 / len(X) * (a_o - y)
    da_od_zo = a_o * (1 - a_o)
    dl_dz_o = dl_da_o * da_od_zo

    # Output weight gradients
    dW_o = h.T @ dl_dz_o

    # Output bias gradient
    db_o = np.sum(dl_dz_o, axis=0)

    # Hidden activation gradients
    dl_dh = dl_dz_o @ W_o.T

    # Hidden ReLU derivative
    dh_dhz = (z_h > 0).astype(float)
    dl_dzh = dl_dh * dh_dhz

    # Hidden weight gradients
    dW_h = X.T @ dl_dzh
    # Hidden bias gradients
    db_h = np.sum(dl_dzh, axis=0)

    # Gradient descent
    W_h = W_h - A * dW_h
    b_h = b_h - A * db_h

    W_o = W_o - A * dW_o
    b_o = b_o - A * db_o
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

z_h_new = X @ W_h + b_h
h_new = np.maximum(0, z_h_new)
z_o_new = h_new @ W_o + b_o
a_o_new = 1 / (1 + np.exp(-z_o_new))

error_new = (a_o_new - y)
squared_error_new = error_new ** 2
new_loss = np.mean(squared_error_new)
predictions = (a_o_new >= 0.5).astype(int)

print("Final probabilities:")
print(a_o_new)

print("XOR predictions:")
print(predictions)

print("Expected:")
print(y.astype(int))

print("Final MSE:", new_loss)
