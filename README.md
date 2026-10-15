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
| [CHEATSHEET.md](CHEATSHEET.md) | Fast formula and terminology reference |
| [examples/hidden_layer_numpy.py](examples/hidden_layer_numpy.py) | Executable NumPy forward pass with assertions |
| [assets/hidden-layer-forward-pass.svg](assets/hidden-layer-forward-pass.svg) | Scalable diagram of calculations inside a hidden layer |
| [assets/hidden-layer-forward-pass.png](assets/hidden-layer-forward-pass.png) | PNG export of the forward-pass diagram |
| [examples/neuron.py](examples/neuron.py) | Pure Python weighted sum, bias, and sigmoid |
| [examples/loss.py](examples/loss.py) | Pure Python loss calculations |

Run the examples with Python 3:

```bash
python examples/neuron.py
python examples/loss.py
python examples/hidden_layer_numpy.py
```

## Source

[3Blue1Brown — But what is a neural network? (Deep learning chapter 1)](https://youtu.be/aircAruvnKk)

## Current status

**Fundamentals:** Neurons, weights, bias, forward propagation, ReLU and sigmoid practiced through 15 October 2026.  
**Loss:** Conceptual introduction only; full training not yet implemented.  
**Next to learn:** loss functions in more depth, gradient descent, backpropagation, and training a small model.

The NumPy forward-pass example has hand-picked weights. **It is not a trained network.**

For matrix arithmetic and broadcasting details, see [ML-MATH](https://github.com/Vaibh37/ML-MATH/tree/main/linear-algebra/matrices).
