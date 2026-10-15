# 02 — Training and Loss

## 1. What is training?

In the earlier examples, we selected weights and biases ourselves. During **training**, a neural network adjusts these values using examples.

For handwritten digit recognition:

1. Give the network an image with a known label, such as `7`.
2. Compute the network's outputs (a **forward pass**).
3. Compare its outputs to the correct target using a **loss function**.
4. Use an optimization method to change weights and biases in a direction intended to reduce the loss.
5. Repeat using more examples.

We have learned the **idea** of training, but we have not yet implemented the method for calculating the parameter updates.

## 2. What is loss?

A **loss function** turns the difference between a prediction and the desired answer into a number to minimize.

For a simple numeric prediction, one example is **squared error**:

```python
actual = 8
predicted = 4

loss = (actual - predicted) ** 2
print(loss)  # 16
```

Python's `** 2` means **square**, not multiply by two.

The squared error is:

`(8 - 4)² = 4² = 16`

## 3. Compare two predictions

If the correct value is 10:

| Prediction | Squared error |
| --- | ---: |
| 4 | 36 |
| 7 | 9 |
| 9 | 1 |
| 10 | 0 |

**Smaller loss means the prediction is closer** when comparing squared errors for this same target.

Example question:

- Prediction A has loss 25.
- Prediction B has loss 4.

**B is better according to this loss function**, since 4 is smaller than 25.

## 4. Minimizing loss

The objective of training is to find weights and biases that produce **low loss** on relevant examples.

This is the conceptual cycle:

```text
input data
    ↓
forward pass (prediction)
    ↓
loss calculation
    ↓
parameter update
    ↓
repeat
```

An update does **not guarantee** that every individual prediction will improve. Low training loss also does not necessarily mean the network will work well on unseen data.

## 5. Important distinction for digit recognition

The scalar squared-error examples above predict a **number** such as 80 or 10. Digit recognition typically uses **10 outputs**, one for each digit. A classification loss such as **cross-entropy** is common for that task.

We have **not** studied cross-entropy, gradient descent, or backpropagation yet. Those are upcoming topics, not prerequisites for understanding the basic example here.

## Check yourself

```python
actual = 10
predicted = 7
loss = (actual - predicted) ** 2
```

What is the loss? **9**.

Next exercise:

```python
actual = 8
predicted = 4
loss = (actual - predicted) ** 2
```

What is the loss? **16**.

## Next

Learn how the network can update its parameters using **gradient descent**, then study **backpropagation**. Do not mistake understanding the training loop for having implemented those methods.

## 6. New checkpoint: NumPy MSE and linear prediction (15 October 2026)

The learner **implemented** mean squared error using NumPy arrays: subtract predictions, square elementwise with `** 2`, and take `np.mean()`. For `actual=[10,20,30,40]` and `predicted=[12,18,33,37]`, MSE is **6.5**.

A separate linear prediction example used `y_pred = w * x + b`, with `x=[1,2,3,4]`, targets `[20,40,60,80]`, `w=10`, `b=0`, and MSE **750**. Manually changing `w` to 15 gave MSE **187.5**. This was **not** gradient descent or learned parameter updates.

For derivation of the loss notation and partial gradients, see [05 — From a forward pass to a loss gradient](05-linear-regression-loss-and-gradients.md). Next: implement parameter updates, then backpropagation.
