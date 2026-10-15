# 05 — From a Forward Pass to a Loss Gradient

**Study checkpoint: 15 October 2026.** The examples below have been introduced and their NumPy arithmetic practiced. **Gradient descent and backpropagation are not yet coded.**

## What part of forward propagation is complete?

For one input column $x$, our 3-neuron hidden layer computed:

$$
Z=Wx+b,\qquad A=\operatorname{ReLU}(Z).
$$

With $x=[[2],[-1]]$, $W=[[2,1],[-1,3],[1,-2]]$, and $b=[[1],[2],[-3]]$, the result was $Z=[[4],[-3],[1]]$ and $A=[[4],[0],[1]]$, shape $(3,1)$. These weights were chosen manually. **We have not completed the full 2→3→1 output-layer forward pass or trained the XOR model.**

## 1. Measure the prediction error (MSE)

For $n$ examples, squared errors average to:

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2.
$$

```python
import numpy as np

y_true = np.array([10, 20, 30, 40])
y_pred = np.array([12, 18, 33, 37])

error = y_true - y_pred
squared = error ** 2
mse = np.mean(squared)
print(error)    # [-2  2 -3  3]
print(squared)  # [4 4 9 9]
print(mse)      # 6.5
```

The sign of the error does not affect **squared** error; $\hat y-y$ and $y-\hat y$ have the same squares. However, sign matters when we later differentiate, so use one convention consistently.

**Interpretation:** MSE is minimized at 0 for perfect predictions; it is not accuracy, and we do not want MSE to approach 1.

## 2. One-neuron linear regression: prediction first, learning later

A linear model with input $x$, weight $w$, and bias $b$ makes predictions using:

$$
\hat y=wx+b
$$

The practiced example was:

```python
x = np.array([1, 2, 3, 4])
y_true = np.array([20, 40, 60, 80])

w = 10
b = 0
y_pred = w * x + b
mse = np.mean((y_true - y_pred) ** 2)

print(y_pred)  # [10 20 30 40]
print(mse)     # 750.0
```

By manually changing only $w$ from 10 to 15, the loss fell to 187.5. The model **did not learn that change automatically**; this was an experiment about the loss curve.

With $b=0$ and the above data, the loss as a function of $w$ is:

$$
L(w)=\operatorname{mean}((20x-wx)^2)=7.5(w-20)^2.
$$

Thus $w=20$ gives zero loss. To read the graph: increasing $w$ moves right on the horizontal axis; higher loss is upward. To the left of $w=20$, moving right is downhill. To the right, moving left is downhill.

## 3. What derivative notation asks

$\frac{dL}{dw}$ is the derivative of loss with respect to the weight. $\frac{\partial L}{\partial b}$ asks how loss changes as bias changes while the other independent variables remain fixed. Differentiating is **not moving terms from the equation's RHS to its LHS**.

For our fixed-bias loss curve:

$$
\frac{dL}{dw}=15(w-20).
$$

- Negative derivative (left of 20): a small positive change of $w$ decreases loss.
- Positive derivative (right of 20): a small negative change of $w$ decreases loss.
- Zero derivative at 20: flat at the minimum for this particular convex curve.

The update rule to implement later is $w_{\mathrm{new}}=w-\alpha\frac{dL}{dw}$. Here $\alpha$ is a learning rate. This rule has been explained with numbers but **has not yet been implemented in a training loop**.

## 4. Where does L(w,b) come from?

For one labeled training example with input $x=4$ and target $y=10$:

1. Predict: $\hat y=wx+b=4w+b$.
2. Define one-example squared-error loss: $L=(\hat y-y)^2$.
3. Substitute: $L(w,b)=(4w+b-10)^2$.

The function notation $L(w,b)$ means **loss depends on these two parameters**; the parentheses do not mean multiplication.

Differentiate using the chain rule:

$$
\frac{\partial L}{\partial w}=2(4w+b-10)\cdot4=8(4w+b-10),
$$
$$
\frac{\partial L}{\partial b}=2(4w+b-10)\cdot1=2(4w+b-10).
$$

These are gradients of **one-example squared error**, not an independently trained model or the batch MSE.

## Next checkpoint (16 October)

1. Reinforce the notation and derivatives of a loss, without repeating arithmetic drills.
2. Implement a single gradient descent step for $w$ and $b$ with a learning rate.
3. Build a small linear-regression training loop and check that loss decreases.
4. Then derive activation gradients, backpropagation, and train the 2→3→1 XOR model.

Full differentiation notes: [ML-MATH calculus](https://github.com/Vaibh37/ML-MATH/tree/main/calculus). Reproducible **prediction and loss only**: [example](../examples/linear_regression_and_mse.py).
