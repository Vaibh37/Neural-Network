# 01 — Neural Network Fundamentals

These notes start with numbers, not advanced mathematics.

## 1. Input and output

An **input** is a number (or collection of numbers) given to a program. An **output** is the result.

```python
x = 7
output = x + 3
print(output)  # 10
```

## 2. Pixels and input neurons

A grayscale image is made of pixels. In our example, pixel brightness is normalized:

- `0.0` means black.
- `0.5` means grey.
- `1.0` means white.

A 28 × 28 image has **784 pixels**. A corresponding input layer can hold **784 input values**, one for each pixel. An **input neuron** here simply represents one of those input numbers.

```python
pixels = [0.2, 0.7, 0.9]  # Toy image with three pixel values
```

## 3. Hidden neurons

A hidden neuron reads numbers from the previous layer, processes them, and gives an output for the next layer. Think of one neuron as a small calculator.

## 4. Weights

A **weight** controls the effect of an input on a neuron's calculation. Each connection to a neuron can have a different weight.

```python
x = 5
weight = 2
weighted_input = x * weight  # 10
```

Weights may be positive, negative, or zero.

## 5. Weighted sum

Multiply each input by its own weight; then add the products.

```python
x1, x2 = 2, 3
w1, w2 = 4, 5
weighted_sum = x1 * w1 + x2 * w2
print(weighted_sum)  # 23
```

## 6. Bias: what it is and why it helps

A **bias** is an extra number added to the weighted sum:

```python
x = 3
weight = 2
bias = 4
z = x * weight + bias  # 10
```

**Why use it?** The bias shifts when a neuron responds strongly or crosses a decision threshold.

For example, suppose a neuron is considered *on* when `z > 0`:

```python
score = 3
bias = -5
z = score + bias
is_on = z > 0       # False
```

Here `score + (-5) > 0` only when `score > 5`. The bias lets us place the threshold at 5 instead of 0 (for weight 1).

The values of weights and biases are usually **learned during training**; the numbers in these notes are hand-picked examples.

## 7. Activation function

The neuron first calculates `z = weighted_sum + bias`. An **activation function** then transforms `z` into the neuron's output.

An easy example is a step function:

```python
z = -3
activation = 1 if z > 0 else 0  # 0
```

Not all activation functions return only 0 or 1.

## 8. Sigmoid

Sigmoid returns a value **strictly between 0 and 1** for any finite input:

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

print(sigmoid(0))   # 0.5
print(sigmoid(10))  # About 0.99995
```

Examples: negative inputs map below 0.5; zero maps to 0.5; positive inputs map above 0.5.

## 9. Layers and connections

A **layer** is a group of neurons:

- **Input layer:** carries input values.
- **Hidden layers:** process values.
- **Output layer:** provides final outputs.

Connections carry values forward and have weights. A network can have multiple hidden layers.

In the 3Blue1Brown example, digit recognition uses **784 inputs**, **two hidden layers**, and **10 output neurons**, one per digit from 0 to 9. The output with the highest activation is chosen as the predicted digit.

## 10. Put it together: forward calculation

This tiny example uses hand-picked weights and biases. Activation functions are omitted here to make the arithmetic easy.

```python
input_number = 2
hidden = input_number * 3 + 1  # 7
output = hidden * 2            # 14
print(output)
```

A more realistic single-neuron calculation is:

```python
import math

x1, x2 = 2.0, 3.0
w1, w2 = 4.0, 5.0
bias = -1.0
z = (x1 * w1) + (x2 * w2) + bias  # 22
activation = 1 / (1 + math.exp(-z))
```

**Mental model:** inputs → multiply by weights → sum → add bias → apply activation → pass output to the next layer.

## Check yourself

1. How many input values are in a 28 × 28 grayscale image?
2. What does a weight do?
3. What does the bias change in a simple threshold neuron?
4. What does sigmoid output when its input is 0?
5. If `(2 * 3) + (4 * 2)` is a weighted sum, what is its value?

Answers: 784; input influence; the threshold; 0.5; 14.
