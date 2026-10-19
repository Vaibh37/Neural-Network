# Neural Network

Built a 2 → 3 → 1 neural network in NumPy while learning backpropagation. The idea was to understand the gradients and weight updates by writing them out myself, without PyTorch, TensorFlow, or autograd.

It uses ReLU in the hidden layer, sigmoid at the output, MSE for loss, and full-batch gradient descent.

## Run

```bash
python -m pip install -r requirements.txt
python examples/xor_network.py
```

## XOR results

Trained for 5,000 epochs with learning rate `0.05` and random seed `42`.

```text
Input   Target   Output
0, 0      0      0.0839
0, 1      1      0.9505
1, 0      1      0.9517
1, 1      0      0.0839
```

Final MSE: `0.004715`. All four inputs classify correctly.

These are training results on the four XOR examples, not a test on unseen data. I haven't checked every gradient numerically yet.

## Files

- [XOR network](examples/xor_network.py) — training code
- [Notes](notes/) — my math notes, from single neurons to hidden-layer backpropagation
- [Cheat sheet](CHEATSHEET.md) — equations and NumPy shapes
- [Diagrams](assets/xor/README.md) — XOR visuals and editable Excalidraw files

I also wrote about the math and training process [on Medium](https://medium.com/@devvaibhav37/building-a-neural-network-from-scratch-the-mathematics-behind-learning-xor-10441e674a69).
