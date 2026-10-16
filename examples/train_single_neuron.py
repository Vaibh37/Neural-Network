"""First gradient descent training loop: one example, one linear neuron.

No dependencies required. Run: python examples/train_single_neuron.py
This is a learning example, not the planned XOR network.
"""

x = 2.0
y = 10.0  # training target

w = 3.0
b = 1.0
learning_rate = 0.05

initial_prediction = w * x + b
initial_loss = (initial_prediction - y) ** 2
print(f"Initial: prediction={initial_prediction:.5f}, loss={initial_loss:.8f}")

for epoch in range(5):
    # Step 1: predict with the CURRENT parameters.
    pred_y = w * x + b

    # Step 2: measure the error before updating.
    loss_before = (pred_y - y) ** 2

    # Step 3: chain-rule gradients of single-example squared error.
    dw = 2 * (pred_y - y) * x
    db = 2 * (pred_y - y)

    # Step 4: replace w and b with their updated values.
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Step 5: measure the result AFTER updating (not used for gradients above).
    pred_y = w * x + b
    loss = (pred_y - y) ** 2
    assert loss < loss_before

    print(
        f"Epoch {epoch + 1}: w={w:.5f}, b={b:.5f}, "
        f"prediction={pred_y:.5f}, loss={loss:.8f}"
    )

# The loop itself does not return a value. The variables still hold the
# values assigned by the fifth update.
assert abs(pred_y - 9.90625) < 1e-10
assert abs(loss - 0.0087890625) < 1e-10
print("Checks passed.")
