"""Regression check for the seeded XOR training example.

This checks training-set behavior only; it makes no generalization claim.
"""
from __future__ import annotations

import contextlib
import io
import runpy
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "xor_network.py"


def main() -> None:
    # Suppress the normal tutorial log while still running its original code.
    with contextlib.redirect_stdout(io.StringIO()):
        variables = runpy.run_path(str(SOURCE), run_name="__main__")

    predictions = variables["predictions"]
    probabilities = variables["a_o_new"]
    labels = variables["y"].astype(int)
    loss = float(variables["new_loss"])

    assert predictions.shape == labels.shape == (4, 1)
    assert probabilities.shape == (4, 1)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_array_equal(predictions, labels)
    assert 0.0 <= loss < 0.01, f"Unexpected final MSE: {loss}"
    print(f"PASS: 4/4 XOR training labels, final MSE {loss:.6f}")


if __name__ == "__main__":
    main()
