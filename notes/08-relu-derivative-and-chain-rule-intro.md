# 08 — ReLU Derivative and Chain Rule: First Introduction (16 October 2026)

**Status:** Practiced elementwise ReLU derivative with NumPy; introduced the two-factor chain rule through ReLU. **Not yet implemented:** a full weight gradient through activation, hidden-layer backpropagation, or XOR training.

## Notation first

| Symbol | Meaning |
| --- | --- |
| w | Weight, a trainable parameter |
| b | Bias, a trainable parameter |
| z | Pre-activation (z = wx + b) |
| a | Activation output (a = ReLU(z)) |
| y | Target |
| L | Loss, here (a-y)^2 |
| ∂L/∂a | How loss changes when a changes |
| ∂a/∂z | How activation changes when z changes |
| ∂L/∂z | How loss changes when z changes, **through a** |

Forward dependency: **w -> z -> a -> L**. The backpropagation calculation follows this path in reverse.

## ReLU and its derivative

~~~math
a=\mathrm{ReLU}(z)=\max(0,z)
~~~

~~~math
\frac{\partial a}{\partial z}=
\begin{cases}
0 & z<0\\
1 & z>0
\end{cases}
~~~

At **z = 0** the derivative is mathematically undefined. A practical convention is to use 0.

~~~python
z = np.array([-3.0, 2.0, 0.0, 5.0])
mask = z > 0                          # [False, True, False, True]
dz = mask.astype(float)               # [0., 1., 0., 1.]
~~~

- z > 0 compares every element of the array; a Python if statement cannot take this whole Boolean array as one truth value.
- .astype(float) converts Boolean True to 1.0 and False to 0.0.
- == tests equality; = assigns a value.

See [runnable ReLU derivative example](../examples/relu_derivative_numpy.py).

## Where the chain-rule equation comes from

We have:

~~~math
z=wx+b,\qquad a=\mathrm{ReLU}(z),\qquad L=(a-y)^2.
~~~

To find how **z** affects **L**, we must go through the intermediate variable **a**:

~~~math
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}.
~~~

This is **the chain rule**, not a new rule invented for neural networks. A small change in z causes approximately (∂a/∂z) times that change in a. That change in a causes approximately (∂L/∂a) times that amount in L. The rates multiply along the same dependency path.

Differentiate the squared-error loss with respect to a:

~~~math
\frac{\partial L}{\partial a}=2(a-y).
~~~

Differentiate a = ReLU(z) with respect to z: ∂a/∂z is ReLU'(z).

**Worked example:** z = -3, y = 2 gives a = 0, L = 4, ∂L/∂a = -4, ∂a/∂z = 0. Therefore ∂L/∂z = (-4) * 0 = **0**. ReLU blocks the local gradient for negative pre-activation.

If an earlier example uses an "incoming gradient" of 6, **6 is simply an assumed value of ∂L/∂a** for illustration; it does not come from differentiating this specific squared-error example.

For positive z, ReLU's local derivative is 1, so it passes the incoming gradient unchanged (at this activation).

## Learning boundary

Understanding why the chain rule multiplies has begun to click, but **do not mark multi-layer backpropagation as completed**. The next lesson derives ∂L/∂w and ∂L/∂b along w -> z -> a -> L, then applies the idea to multiple neurons.
