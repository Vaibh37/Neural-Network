"""Reproduce the MSE and linear-regression prediction exercises from 15 Oct 2026.

This script computes predictions and losses only. It does not train parameters.
"""
import numpy as np


# Exercise 1: MSE from supplied predictions.
actual = np.array([10, 20, 30, 40])
predicted = np.array([12, 18, 33, 37])
errors = actual - predicted
squared_errors = errors ** 2
mse = np.mean(squared_errors)

print("errors:", errors)                    # [-2  2 -3  3]
print("squared errors:", squared_errors)    # [4 4 9 9]
print("MSE:", mse)                          # 6.5
assert np.isclose(mse, 6.5)


# Exercise 2: One-neuron linear prediction with hand-picked w and b.
x = np.array([1, 2, 3, 4])
y_true = np.array([20, 40, 60, 80])
b = 0

for w, expected_mse in [(10, 750.0), (15, 187.5), (20, 0.0)]:
    y_pred = w * x + b
    loss = np.mean((y_true - y_pred) ** 2)
    print(f"w={w}, predictions={y_pred}, MSE={loss}")
    assert np.isclose(loss, expected_mse)

# No gradient or parameter update is performed.
