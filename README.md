# Neural-Network

Personal notes and runnable Python examples for learning neural network fundamentals from first principles.

## Topics covered

- Inputs, outputs, pixels, and input neurons
- Hidden neurons, weights, weighted sums, and biases
- Activation functions, including sigmoid
- Layers, connections, and forward calculations
- Training at a high level: predictions, error, and loss
- Squared error and the goal of minimizing loss

## Learning material

| Path | Content |
| --- | --- |
| [notes/01-fundamentals.md](notes/01-fundamentals.md) | Neurons, layers, weights, biases, sigmoid, and a complete forward-pass example |
| [notes/02-training-and-loss.md](notes/02-training-and-loss.md) | How training works conceptually, squared error, and loss minimization |
| [examples/neuron.py](examples/neuron.py) | Pure Python weighted sum, bias, and sigmoid |
| [examples/loss.py](examples/loss.py) | Pure Python loss calculations |

Run the examples with Python 3:

```bash
python examples/neuron.py
python examples/loss.py
```

## Source

[3Blue1Brown — But what is a neural network? (Deep learning chapter 1)](https://youtu.be/aircAruvnKk)

## Current status

**Fundamentals and loss:** reviewed.  
**Next to learn:** how optimization changes weights and biases; gradient descent and backpropagation.

These are learning notes, not an implementation of a trained digit classifier.
