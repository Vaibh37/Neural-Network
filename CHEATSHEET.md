# Neural Network Cheat Sheet

Equations and NumPy shapes used in the [XOR implementation](examples/xor_network.py). For worked derivations, see the [notes](notes/).

## One neuron

For inputs `x`, weights `w`, bias `b`, and activation `f`:

~~~math
z=w\cdot x+b,\qquad a=f(z)
~~~

- `z` is the weighted sum before activation.
- `a` is the output after activation.
- ReLU: `max(0, z)`.
- Sigmoid: `1 / (1 + exp(-z))`.
- ReLU derivative: 1 if `z > 0`, 0 if `z < 0`. At `z = 0`, the derivative is undefined; the implementation uses 0.

For a **linear** neuron (`a = wx + b`) with squared-error loss `L = (a - y)²`:

~~~math
\frac{\partial L}{\partial w}=2(a-y)x,\qquad
\frac{\partial L}{\partial b}=2(a-y)
~~~

For a nonlinear activation, apply the chain rule through `a = f(z)` before finding the weight and bias gradients.

## XOR network: shapes

The code stores **one training example per row**. There are four examples (`N = 4`).

| Variable | Meaning | Shape |
| --- | --- | --- |
| `X` | Input batch | `(4, 2)` |
| `y` | Targets | `(4, 1)` |
| `W_h`, `b_h` | Hidden weights and biases | `(2, 3)`, `(3,)` |
| `z_h`, `h` | Hidden pre-activations and ReLU outputs | `(4, 3)` |
| `W_o`, `b_o` | Output weights and bias | `(3, 1)`, `(1,)` |
| `z_o`, `a_o` | Output pre-activations and sigmoid outputs | `(4, 1)` |
| `L` | Mean squared error | Scalar |

There are **13 parameters**: 6 hidden weights + 3 hidden biases + 3 output weights + 1 output bias.

## Forward pass

The hidden layer receives `X`:

~~~math
Z_h=XW_h+b_h,\qquad H=\mathrm{ReLU}(Z_h)
~~~

The output layer receives `H`:

~~~math
Z_o=HW_o+b_o,\qquad A_o=\sigma(Z_o)
~~~

~~~python
z_h = X @ W_h + b_h
h = np.maximum(0, z_h)

z_o = h @ W_o + b_o
a_o = 1 / (1 + np.exp(-z_o))
~~~

## Loss

With `N` examples and one output per example:

~~~math
L=\frac{1}{N}\sum_{i=1}^{N}(a_o^{(i)}-y^{(i)})^2
~~~

~~~python
loss = np.mean((a_o - y) ** 2)
~~~

## Backpropagation

Find every gradient **before** updating any parameter.

**1. Loss → sigmoid output → output weighted sum**

~~~math
\frac{\partial L}{\partial A_o}=\frac{2}{N}(A_o-y)
~~~

~~~math
\frac{\partial L}{\partial Z_o}
=\frac{\partial L}{\partial A_o}\odot A_o\odot(1-A_o)
~~~

~~~python
dl_da_o = 2 / len(X) * (a_o - y)
da_od_zo = a_o * (1 - a_o)
dl_dz_o = dl_da_o * da_od_zo
~~~

The `2/N` factor is already included. Do not average the parameter gradients again.

**2. Output parameters**

~~~python
dW_o = h.T @ dl_dz_o         # (3, 1)
db_o = np.sum(dl_dz_o, axis=0)  # (1,)
~~~

**3. Output → hidden activations → hidden weighted sum**

~~~math
\frac{\partial L}{\partial H}
=\frac{\partial L}{\partial Z_o}W_o^T
~~~

~~~python
dl_dh = dl_dz_o @ W_o.T
relu_mask = (z_h > 0).astype(float)
dl_dzh = dl_dh * relu_mask
~~~

The ReLU mask is elementwise; `W_o.T` is the **weight matrix transpose**, not the gradient transpose.

**4. Hidden parameters**

~~~python
dW_h = X.T @ dl_dzh          # (2, 3)
db_h = np.sum(dl_dzh, axis=0)  # (3,)
~~~

Summing with `axis=0` combines the four examples but keeps a separate bias gradient per neuron.

## Gradient descent

For each parameter `θ` with gradient `∂L/∂θ` and learning rate `α`:

~~~math
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\alpha\frac{\partial L}{\partial\theta}
~~~

~~~python
W_h -= A * dW_h
b_h -= A * db_h
W_o -= A * dW_o
b_o -= A * db_o
~~~

The XOR script uses seed `42`, learning rate `0.05`, and `5,000` full-batch updates. The reported final MSE is about `0.004715`, with correct classifications for all four **training** examples. This is not a held-out evaluation. Numerical gradient checking has not yet been implemented.

## Worked lessons

- [Single-neuron gradient descent](notes/06-first-gradient-descent-training-loop.md)
- [Chain rule and parameter gradients](notes/09-chain-rule-and-backpropagation-gradients.md)
- [XOR forward pass and MSE](notes/12-xor-vectorized-forward-and-loss.md)
- [Output-layer backpropagation](notes/13-xor-output-layer-backprop.md)
- [Hidden-layer backpropagation](notes/14-xor-hidden-layer-backprop.md)
- [Training loop and parameter updates](notes/15-xor-gradient-descent-and-epochs.md)
- [XOR diagrams](assets/xor/README.md)
