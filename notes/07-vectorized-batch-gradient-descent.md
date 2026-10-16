# 07 — Vectorization and Full-Batch Gradient Descent (16 October 2026)

**Status:** Implemented and practiced: four training examples, one linear neuron, vectorized NumPy predictions, dataset MSE, analytic gradients, and a five-epoch full-batch training loop. This is **not** hidden-layer backpropagation.

## The dataset and the shared parameters

Inputs and targets are 1D NumPy arrays of shape (4,):

~~~python
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 10.0, 15.0, 20.0])
w, b = 3.0, 1.0
~~~

There are **four examples but only one neuron**, with a **single shared weight w and bias b**. No second neuron is involved in this training loop.

Each example is predicted using the same equation:

~~~math
\hat y_i = wx_i + b
~~~

NumPy broadcasts scalar w and b over the array x:

~~~python
pred_y = w * x + b      # [4., 7., 10., 13.]; shape (4,)
~~~

## MSE: square before taking the mean

~~~math
L(w,b) = \frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
~~~

- L = mean squared error (dataset loss)
- n = number of examples, here 4
- i = index of one example
- y_i = target; y-hat_i = prediction

~~~python
squared_errors = (pred_y - y) ** 2    # [1., 9., 25., 49.]
loss = np.mean(squared_errors)          # 21.0
~~~

**Common mistake:** np.mean(pred_y - y) ** 2 squares the *mean error*, which is not MSE. Square each error first, then take the mean.

## Where the dataset gradients come from

For **one** example the squared-error gradients are:

~~~math
\frac{\partial L_i}{\partial w}=2(\hat y_i-y_i)x_i,\qquad
\frac{\partial L_i}{\partial b}=2(\hat y_i-y_i).
~~~

The derivative with respect to w leaves x_i as the constant multiplier; the derivative with respect to b uses an inner derivative of 1.

For the **mean loss** over n examples, average the per-example gradients:

~~~math
\frac{\partial L}{\partial w}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)x_i,\qquad
\frac{\partial L}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i).
~~~

In this lesson, Python names these **gradients** dw and db:

~~~python
dw = 2 * np.mean((pred_y - y) * x)  # -25.0 initially
db = 2 * np.mean(pred_y - y)        #  -8.0 initially
~~~

The original per-example weight gradient array is [-2., -12., -30., -56.]. Its mean is -25.0. This is a **single scalar** because there is only one weight. With multiple weights, gradients have matching parameter shapes; the averaging is over samples, not over different parameters.

**Common mistake:** np.mean(pred_y - y) * x calculates the mean *before* multiplying each error by its input. It does not produce the correct dataset weight gradient.

## Why calculate gradients?

- w and b are the **parameters we update**.
- dw = partial derivative of L with respect to w; db = partial derivative of L with respect to b. They measure how loss changes with each parameter.
- Gradients are **computed from current predictions**, not trained or updated as parameters themselves.
- The learning rate alpha controls how far parameters move.

~~~math
w_{\rm new}=w-\alpha\frac{\partial L}{\partial w},
\qquad b_{\rm new}=b-\alpha\frac{\partial L}{\partial b}.
~~~

For alpha = 0.05 on the first batch: w changes 3 -> 4.25, b changes 1 -> 1.4. Initial MSE is 21, and post-update MSE is 0.92875.

**Important:** Use db when updating b, not dw. Also, lower loss after an update does not prove the gradients are implemented correctly.

## Five epochs: full-batch training

One epoch means:
1. Predict all four examples with current w and b.
2. Compute dw and db from all four examples.
3. Update the **same** shared w and b once.
4. Recompute predictions and MSE for reporting.
5. Repeat with the new parameters.

| After epoch | Post-update MSE |
| ---: | ---: |
| Initial | 21.000000 |
| 1 | 0.928750 |
| 2 | 0.371897 |
| 3 | 0.346442 |
| 4 | 0.335756 |
| 5 | 0.325768 |

These values are for the specified initial parameters and learning rate. Five epochs reduce loss but do not perfectly fit the four examples. A falling training loss does not establish generalization.

**Run and verify:** [examples/train_batch_neuron_numpy.py](../examples/train_batch_neuron_numpy.py)

~~~bash
python examples/train_batch_neuron_numpy.py
~~~

## Next

Derivative of ReLU; the chain rule on the path **w -> z -> a -> L**; then hidden-layer backpropagation and a trainable 2 -> 3 -> 1 XOR network.
