# Neural Network Fundamentals — Cheat Sheet

**Checkpoint:** 15 October 2026. This is a **forward-pass reference**, not yet a model-training guide.

## One neuron

```math
z=w\cdot x+b,\quad a=f(z).
```

- **x:** numeric input vector.
- **w:** weights (one per input).
- **b:** bias added to the weighted sum.
- **z:** pre-activation value.
- **a:** activated output sent to the next layer.

For x = [2, 3], w = [1, -1], b = 1:

z = (1 × 2) + (-1 × 3) + 1 = **0**. Sigmoid(0) = **0.5**.

## Dense layer (multiple neurons)

```math
Z=WX+b,\qquad A=f(Z).
```

With **one column input**, W:(3,2), x:(2,1), b:(3,1) gives Z:(3,1), A:(3,1).

With **s samples stored as columns**, X:(2,s), the same W and b give output A:(3,s). The bias is broadcast across sample columns.

| Operation | What it does |
|---|---|
| W @ X | Matrix product: row × column |
| + b | Adds one bias per output neuron |
| ReLU(Z) | max(0,z) for every entry |
| sigmoid(Z) | 1 / (1 + e^(-z)) for every entry |
| A | Activated outputs passed to the next layer |

**Warning:** An activated dense layer is often described as one neural-network layer even when code defines its linear operation and activation as separate modules. They are not inherently two groups of neurons.

## Activation reference

| Input | ReLU | Sigmoid (approx.) |
|---:|---:|---:|
| -5 | 0 | 0.0067 |
| 0 | 0 | 0.5 |
| 3 | 3 | 0.9526 |
| 10 | 10 | 0.99995 |

ReLU commonly appears in hidden layers; sigmoid can map a binary classifier logit to a number in (0,1). Not every sigmoid output is a calibrated probability.

## Full practice example

```math
x=\begin{bmatrix}2\\-1\end{bmatrix},\quad
W=\begin{bmatrix}2&1\\-1&3\\1&-2\end{bmatrix},\quad
b=\begin{bmatrix}1\\2\\-3\end{bmatrix}.
```

```math
Z=Wx+b=\begin{bmatrix}4\\-3\\1\end{bmatrix},
\quad
A=\mathrm{ReLU}(Z)=\begin{bmatrix}4\\0\\1\end{bmatrix}.
```

Output shape: **(3,1)**. See [step-by-step lesson](notes/04-hidden-layer-forward-pass.md) and [runnable example](examples/hidden_layer_numpy.py).

## What we have *not* implemented

Backpropagation, gradient calculations, optimizer updates, training loops, evaluation, or a trained digit classifier. A successful forward pass is **not** training.

## MSE and linear prediction — practiced 15 October 2026

- Prediction for one feature: `y_pred = w * x + b` (elementwise for a NumPy array of samples).
- Squared error: `(y_pred - y_true) ** 2` (one error per sample).
- Mean squared error: `np.mean((y_pred - y_true) ** 2)` (one scalar average).
- MSE of **0** means perfect predictions on the evaluated examples; unlike accuracy, a larger value is worse.
- In the practiced dataset `x=[1,2,3,4]`, `y_true=[20,40,60,80]`, `b=0`: at `w=10`, MSE **750**; at `w=15`, **187.5**. These weights were modified manually.
- For one sample `x=4`, target `y=10`: `L(w,b) = (4w+b-10)^2`. This names a loss function that depends on `w` and `b`; it is **not** a product `L*w*b`.
- Partial derivatives for that one-sample loss: `∂L/∂w = 8(4w+b-10)`, `∂L/∂b = 2(4w+b-10)`.

Full explanation: [linear regression, loss and gradients](notes/05-linear-regression-loss-and-gradients.md). Runnable example: [prediction and loss only](examples/linear_regression_and_mse.py).

**Not done:** trained parameter updates, full output-layer/XOR forward pass, chain rule through a network, or backpropagation.
