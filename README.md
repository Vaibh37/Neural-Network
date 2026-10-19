# XOR Neural Network from Scratch

A small neural network built with Python and NumPy to learn XOR. The forward pass, backpropagation, and gradient descent are implemented directly, without a deep-learning framework or automatic differentiation.

![2–3–1 network architecture](assets/xor/d02-parameter-map.svg)

## Implementation

The network has **2 inputs, 3 ReLU hidden neurons, and 1 sigmoid output** (13 trainable parameters).

```text
X (4, 2)
  ↓
Hidden layer: 3 neurons + ReLU
  ↓
Output layer: 1 neuron + sigmoid
  ↓
Predictions (4, 1)
```

The forward pass computes:

```python
z_h = X @ W_h + b_h
h = np.maximum(0, z_h)

z_o = h @ W_o + b_o
a_o = 1 / (1 + np.exp(-z_o))
```

Training uses mean squared error (MSE). I calculate the gradients for both layers with the chain rule and update the weights and biases using full-batch gradient descent.

- **Dataset:** all four XOR input combinations
- **Initialization:** NumPy random seed `42`, zero biases
- **Learning rate:** `0.05`
- **Training:** `5,000` epochs

**Code:** [examples/xor_network.py](examples/xor_network.py)

## Results

| Input | Expected | Output probability | Predicted |
| --- | ---: | ---: | ---: |
| (0, 0) | 0 | 0.083898 | 0 |
| (0, 1) | 1 | 0.950516 | 1 |
| (1, 0) | 1 | 0.951739 | 1 |
| (1, 1) | 0 | 0.083932 | 0 |

**Final training MSE:** `0.004715`  
**Training-set accuracy:** `4/4`

![Training loss](assets/xor/d11-loss-curve.png)

These results are from the four XOR training examples, not a separate test set. Numerical gradient checking has not been added yet.

## Run

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python examples/xor_network.py
```

The script prints training loss every 500 epochs, followed by the final probabilities, predicted classes, and MSE.

## Notes and diagrams

I kept the maths and diagrams separate from the main README:

- [XOR architecture and problem](notes/11-xor-problem-and-architecture.md)
- [Vectorized forward pass and loss](notes/12-xor-vectorized-forward-and-loss.md)
- [Output-layer backpropagation](notes/13-xor-output-layer-backprop.md)
- [Hidden-layer backpropagation](notes/14-xor-hidden-layer-backprop.md)
- [Gradient descent and training loop](notes/15-xor-gradient-descent-and-epochs.md)
- [Initialization and training results](notes/16-xor-random-initialization-and-results.md)
- [Diagrams, including editable Excalidraw files](assets/xor/README.md)

Earlier lessons and smaller examples are also available in [notes](notes/) and [examples](examples/).

For a longer explanation, see my [Medium write-up on building this XOR network](https://medium.com/@devvaibhav37/building-a-neural-network-from-scratch-the-mathematics-behind-learning-xor-10441e674a69).
