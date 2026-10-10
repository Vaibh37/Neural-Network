"""A tiny squared-error example, not a digit classification model."""


def squared_error(actual: float, predicted: float) -> float:
    """Measure one numeric prediction using a squared difference."""
    return (actual - predicted) ** 2


if __name__ == "__main__":
    actual = 10

    for predicted in [4, 7, 9, 10]:
        loss = squared_error(actual, predicted)
        print(f"actual={actual}, predicted={predicted}, loss={loss}")

    # A separate practice example: (8 - 4) ** 2 = 16.
    print(f"practice loss = {squared_error(8, 4)}")
