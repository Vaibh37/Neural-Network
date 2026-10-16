"""ReLU derivative: elementwise Boolean comparison and dtype conversion.

Run: python examples/relu_derivative_numpy.py
"""
import numpy as np

z = np.array([-3.0, 2.0, 0.0, 5.0])

# z > 0 returns [False, True, False, True].
# .astype(float) converts False -> 0.0 and True -> 1.0.
relu_derivative = (z > 0).astype(float)
print(relu_derivative)
assert np.array_equal(relu_derivative, [0.0, 1.0, 0.0, 1.0])

# ReLU is not differentiable at exactly zero. We choose 0.0 in code.
