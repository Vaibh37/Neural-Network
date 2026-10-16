# Neural-Network

Personal notes and runnable Python examples for learning neural network fundamentals from first principles.

## Topics covered

- Inputs, outputs, pixels, and input neurons
- Hidden neurons, weights, weighted sums, and biases
- Activation functions, including ReLU and sigmoid
- Layers, connections, and forward calculations
- Training at a high level: predictions, error, and loss
- Squared error and the goal of minimizing loss

## Learning material

| Path | Content |
| --- | --- |
| [notes/01-fundamentals.md](notes/01-fundamentals.md) | Neurons, layers, weights, biases, sigmoid, and a complete forward-pass example |
| [notes/02-training-and-loss.md](notes/02-training-and-loss.md) | How training works conceptually, squared error, and loss minimization |
| [notes/03-activation-functions.md](notes/03-activation-functions.md) | Why activations matter; worked ReLU and sigmoid examples |
| [notes/04-hidden-layer-forward-pass.md](notes/04-hidden-layer-forward-pass.md) | Detailed 2-neuron and 3-neuron forward passes, dimensions and diagrams |
| [notes/05-linear-regression-loss-and-gradients.md](notes/05-linear-regression-loss-and-gradients.md) | Practiced NumPy MSE, linear prediction, derivative notation and single-sample loss gradients |
| [notes/07-vectorized-batch-gradient-descent.md](notes/07-vectorized-batch-gradient-descent.md) | Four-example NumPy gradients, full-batch training, five-epoch loss trace and common mistakes |
| [notes/08-relu-derivative-and-chain-rule-intro.md](notes/08-relu-derivative-and-chain-rule-intro.md) | ReLU derivative, Boolean masks, and first chain-rule step through an activation |
| [examples/train_batch_neuron_numpy.py](examples/train_batch_neuron_numpy.py) | Executable 4-example training loop with numeric checks |
| [examples/relu_derivative_numpy.py](examples/relu_derivative_numpy.py) | Executable ReLU derivative with NumPy |
| [examples/linear_regression_and_mse.py](examples/linear_regression_and_mse.py) | Reproducible prediction and MSE checks; no parameter training |
| [CHEATSHEET.md](CHEATSHEET.md) | Fast formula and terminology reference |
| [examples/hidden_layer_numpy.py](examples/hidden_layer_numpy.py) | Executable NumPy forward pass with assertions |
| [assets/hidden-layer-forward-pass.svg](assets/hidden-layer-forward-pass.svg) | Scalable diagram of calculations inside a hidden layer |
| [assets/hidden-layer-forward-pass.png](assets/hidden-layer-forward-pass.png) | PNG export of the forward-pass diagram |
| [examples/neuron.py](examples/neuron.py) | Pure Python weighted sum, bias, and sigmoid |
| [examples/loss.py](examples/loss.py) | Pure Python loss calculations |

Use Python 3.10+ and install the only third-party dependency before running the examples:

~~~bash
python -m pip install -r requirements.txt
~~~

Then run:

```bash
python examples/neuron.py
python examples/loss.py
python examples/hidden_layer_numpy.py
python examples/linear_regression_and_mse.py
python examples/train_single_neuron.py
python examples/train_batch_neuron_numpy.py
python examples/relu_derivative_numpy.py
```

## Source

[3Blue1Brown — But what is a neural network? (Deep learning chapter 1)](https://youtu.be/aircAruvnKk)

## Current status

**Checkpoint: 16 October 2026.** Single-example plain-Python training and vectorized **four-example, five-epoch full-batch NumPy training** are implemented. For x = [1,2,3,4], y = [5,10,15,20], w = 3, b = 1, and learning rate 0.05, dataset MSE falls from **21.0 to approximately 0.325768**.

**Activation derivatives:** ReLU'(z) with a NumPy Boolean mask and .astype(float) is practiced. The chain rule for **z -> a -> L** is introduced and derived in notes, but full weight gradients through an activation are next.

**Not yet done:** training a hidden layer, full backpropagation through multiple neurons, and the **2 -> 3 -> 1 XOR** network. Target around **20 October 2026**, based on understanding rather than a rigid deadline.

For the precise batch training trace, see [lesson 07](notes/07-vectorized-batch-gradient-descent.md). For ReLU and the chain-rule checkpoint, see [lesson 08](notes/08-relu-derivative-and-chain-rule-intro.md).
