# Neural Network Fundamentals — Cheat Sheet

**Checkpoint:** 16 October 2026. Includes a single-example gradient descent training loop; hidden-layer backpropagation is still upcoming.

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

## Training status

A single linear neuron **has** been trained for five updates on one example, with manual gradients. Backpropagation through hidden layers, multi-example dataset training and a trained digit/XOR classifier **have not** been implemented. A hidden-layer forward pass alone is not training.

## MSE and linear prediction — practiced 15 October 2026

- Prediction for one feature: `y_pred = w * x + b` (elementwise for a NumPy array of samples).
- Squared error: `(y_pred - y_true) ** 2` (one error per sample).
- Mean squared error: `np.mean((y_pred - y_true) ** 2)` (one scalar average).
- MSE of **0** means perfect predictions on the evaluated examples; unlike accuracy, a larger value is worse.
- In the practiced dataset `x=[1,2,3,4]`, `y_true=[20,40,60,80]`, `b=0`: at `w=10`, MSE **750**; at `w=15`, **187.5**. These weights were modified manually.
- For one sample `x=4`, target `y=10`: `L(w,b) = (4w+b-10)^2`. This names a loss function that depends on `w` and `b`; it is **not** a product `L*w*b`.
- Partial derivatives for that one-sample loss: `∂L/∂w = 8(4w+b-10)`, `∂L/∂b = 2(4w+b-10)`.

Full explanation: [linear regression, loss and gradients](notes/05-linear-regression-loss-and-gradients.md). Runnable example: [prediction and loss only](examples/linear_regression_and_mse.py).

**Now done (16 October):** one-example weight/bias parameter updates and a five-epoch Python loop. **Not done:** full XOR network, chain rule through multiple layers, or backpropagation.


## Single-example training equations (16 October)

For one example and a linear output:

~~~math
\begin{aligned}
\hat y&=wx+b, \quad L=(\hat y-y)^2,\\
dw&=2(\hat y-y)x,\quad db=2(\hat y-y),\\
w&\leftarrow w-\alpha\,dw,\quad b\leftarrow b-\alpha\,db.
\end{aligned}
~~~

Repeat with **updated** w and b. A learning rate alpha is chosen and tested; too large can overshoot. Starting with x=2, y=10, w=3, b=1, alpha=0.05: prediction **7→8.5→9.25→9.625→9.8125→9.90625** and loss **9→2.25→0.5625→0.140625→0.03515625→0.0087890625**.

See [full lesson](notes/06-first-gradient-descent-training-loop.md) and [runnable training loop](examples/train_single_neuron.py). **Do not confuse target (10), prediction (initially 7) and loss (initially 9).**
