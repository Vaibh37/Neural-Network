"""A single toy neuron: inputs -> weighted sum -> bias -> sigmoid.

This script demonstrates a forward calculation only. It does not train a model.
"""

import math


def sigmoid(value: float) -> float:
    """Map a number into (0, 1)."""
    return 1.0 / (1.0 + math.exp(-value))


def neuron(inputs: list[float], weights: list[float], bias: float) -> float:
    """Return the sigmoid activation for one neuron."""
    if len(inputs) != len(weights):
        raise ValueError("Each input must have a corresponding weight.")

    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    z = weighted_sum + bias
    print(f"weighted sum = {weighted_sum:.2f}")
    print(f"bias = {bias:.2f}")
    print(f"z = {z:.2f}")
    return sigmoid(z)


if __name__ == "__main__":
    # These numbers are hand-picked to demonstrate the calculation.
    activation = neuron(
        inputs=[0.2, 0.8],
        weights=[0.5, 1.5],
        bias=-0.4,
    )
    print(f"sigmoid output = {activation:.4f}")
