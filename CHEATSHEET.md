# Neural Network Fundamentals — Cheat Sheet

**Checkpoint:** 16 October 2026. Single-example and four-example NumPy batch training practiced; ReLU derivative and chain rule introduced. Hidden-layer backpropagation remains upcoming.

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

A single linear neuron **has** been trained both on one example (plain Python) and on four examples (vectorized NumPy full-batch MSE). Backpropagation through hidden layers and a trained digit/XOR classifier **have not** been implemented. A hidden-layer forward pass alone is not training.

## MSE and linear prediction — practiced 15 October 2026

- Prediction for one feature: `y_pred = w * x + b` (elementwise for a NumPy array of samples).
- Squared error: `(y_pred - y_true) ** 2` (one error per sample).
- Mean squared error: `np.mean((y_pred - y_true) ** 2)` (one scalar average).
- MSE of **0** means perfect predictions on the evaluated examples; unlike accuracy, a larger value is worse.
- In the practiced dataset `x=[1,2,3,4]`, `y_true=[20,40,60,80]`, `b=0`: at `w=10`, MSE **750**; at `w=15`, **187.5**. These weights were modified manually.
- For one sample `x=4`, target `y=10`: `L(w,b) = (4w+b-10)^2`. This names a loss function that depends on `w` and `b`; it is **not** a product `L*w*b`.
- Partial derivatives for that one-sample loss: `∂L/∂w = 8(4w+b-10)`, `∂L/∂b = 2(4w+b-10)`.

Full explanation: [linear regression, loss and gradients](notes/05-linear-regression-loss-and-gradients.md). Runnable example: [prediction and loss only](examples/linear_regression_and_mse.py).

**Now done (16 October):** single-example training; four-example vectorized MSE gradients and a five-epoch batch loop. **Introduced:** ReLU derivative and chain rule through one activation. **Not done:** full XOR network, chain rule through multiple layers, or hidden-layer backpropagation.


## Single-example training equations (16 October)

For one example and a linear output:

```math
\begin{aligned}
\hat y&=wx+b, \quad L=(\hat y-y)^2,\\
dw&=2(\hat y-y)x,\quad db=2(\hat y-y),\\
w&\leftarrow w-\alpha\,dw,\quad b\leftarrow b-\alpha\,db.
\end{aligned}
```
Repeat with **updated** w and b. A learning rate alpha is chosen and tested; too large can overshoot. Starting with x=2, y=10, w=3, b=1, alpha=0.05: prediction **7→8.5→9.25→9.625→9.8125→9.90625** and loss **9→2.25→0.5625→0.140625→0.03515625→0.0087890625**.

See [full lesson](notes/06-first-gradient-descent-training-loop.md) and [runnable training loop](examples/train_single_neuron.py). **Do not confuse target (10), prediction (initially 7) and loss (initially 9).**


## Full-batch NumPy training (16 October)

Four samples but **one shared neuron** (one scalar weight w and bias b):

```python
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 10.0, 15.0, 20.0])
pred_y = w * x + b
L = np.mean((pred_y - y) ** 2)  # Square first, THEN average
dw = 2 * np.mean((pred_y - y) * x)
db = 2 * np.mean(pred_y - y)
w = w - learning_rate * dw
b = b - learning_rate * db
```
- **w** is updated; **dw = ∂L/∂w** is the calculated *weight gradient*, not another weight or a second neuron.
- **b** is updated with **db = ∂L/∂b**, not dw.
- Each example contributes its own error and gradient. Average those contributions before one full-batch parameter update.
- Initial: w = 3, b = 1, learning_rate = 0.05, initial MSE = **21.0**, first dw = **-25**, db = **-8**. After five updates MSE ≈ **0.325768**.
- A gradient must be **recomputed every epoch** from current parameters.

## ReLU derivative and local chain rule

```math
z=wx+b,\quad a=\mathrm{ReLU}(z),\quad L=(a-y)^2
```
```math
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}
=2(a-y)\mathrm{ReLU}'(z)
```
```python
z = np.array([-3.0, 2.0, 0.0, 5.0])
relu_prime = (z > 0).astype(float)  # [0., 1., 0., 1.]
```
ReLU'(z) is 0 for z < 0 and 1 for z > 0; at exactly 0 the derivative is undefined, and the code uses 0 by convention. For z = -3 and y = 2: a = 0, ∂L/∂a = -4, ∂a/∂z = 0, so ∂L/∂z = 0. **Hidden-layer backpropagation remains next.**

Sources: [batch training lesson](notes/07-vectorized-batch-gradient-descent.md), [activation chain rule lesson](notes/08-relu-derivative-and-chain-rule-intro.md).


## One-neuron backpropagation and chain rule (17 October 2026)

**Practiced:** Differentiated the loss, ReLU activation, weighted sum, and combined those gradients using the chain rule. This is a mathematical derivation for one neuron, **not** completed training of the XOR network.

For $z=wx+b$, $a=\mathrm{ReLU}(z)$, and $L=(a-y)^2$:

```math
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w}
```

```math
\frac{\partial L}{\partial b}
=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial b}
```

| Local derivative | Result |
| --- | --- |
| $\partial L/\partial a$ | $2(a-y)$ |
| $\partial a/\partial z$ | $1$ for $z>0$; $0$ for $z<0$; undefined at $z=0$ |
| $\partial z/\partial w$ | $x$ |
| $\partial z/\partial b$ | $1$ |

**Worked example:** $x=2,w=3,b=1,y=10$ gives $z=7,a=7,L=9$. Hence $\partial L/\partial w=(-6)(1)(2)=\mathbf{-12}$ and $\partial L/\partial b=(-6)(1)(1)=\mathbf{-6}$.

![Chain rule cheat sheet](assets/chain-rule/cheat-sheet.svg)

Full derivation: [09 — Chain Rule and Backpropagation Gradients](notes/09-chain-rule-and-backpropagation-gradients.md).


## Two inputs → three hidden neurons → one output (17 October 2026)

**This is one worked XOR sample (x=[1,0], target y=1) using manually chosen weights.** Full hidden-layer training and XOR classification are not completed.

### Hidden-layer forward pass

A neuron with two inputs uses **two weights and one bias**, not the single-input equation wx+b alone:

```math
z_i = w_{i1}x_1+w_{i2}x_2+b_i,\qquad h_i=\mathrm{ReLU}(z_i)
```
With $W_{\mathrm{hidden}}=\begin{bmatrix}1&1\\-1&1\\2&-1\end{bmatrix}$, $x=[1,0]$, and zero hidden biases:

```math
z=[1,-1,2],\quad h=\mathrm{ReLU}(z)=[1,0,2]
```
### Output-layer forward pass

New **independent** output weights $v=[0.2,0.4,0.3]$, and $b_{\mathrm{out}}=0$:

```math
\hat y=v_1h_1+v_2h_2+v_3h_3+b_{\mathrm{out}}=0.8
```
With linear output and **single-example** squared loss, $L=(0.8-1)^2=0.04$.

### Output-layer backpropagation

**Do not multiply the original functions together: multiply the local derivatives along the dependency path.**

```math
\frac{\partial L}{\partial\hat y}=2(\hat y-y)=-0.4
```
```math
\frac{\partial L}{\partial v_i}
=\frac{\partial L}{\partial\hat y}\cdot\frac{\partial\hat y}{\partial v_i}
=2(\hat y-y)h_i
```
```math
\frac{\partial\hat y}{\partial b_{\mathrm{out}}}
=\frac{\partial}{\partial b_{\mathrm{out}}}(v_1h_1+v_2h_2+v_3h_3+b_{\mathrm{out}})
=0+0+0+1=1
```
```math
\frac{\partial L}{\partial b_{\mathrm{out}}}
=\frac{\partial L}{\partial\hat y}\cdot 1=-0.4
```
**Gradients:** $dv=[-0.4,0,-0.8]$, $db_{\mathrm{out}}=-0.4$. During backpropagation **parameters do not change**.

### Gradient descent: update after calculating gradients

With $\eta=0.05$, apply $\theta_{\mathrm{new}}=\theta-\eta\,\partial L/\partial\theta$:

```math
v_{\mathrm{new}}=[0.22,0.40,0.34],\quad b_{\mathrm{out,new}}=0.02
```
New prediction $0.92$; new loss $0.0064$ (**84% less loss on this one example**). Hidden-layer parameters were deliberately **not updated**.

![Output gradient quick reference](assets/lesson-10-output-layer/05-quick-reference.svg)

Full explanation: [Lesson 10 — output-layer forward and backpropagation](notes/10-output-layer-forward-backprop-and-update.md).
Runnable NumPy check: [examples/output_layer_single_step.py](examples/output_layer_single_step.py).
