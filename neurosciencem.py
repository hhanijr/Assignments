import random

def tanh(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1  
w1, w2, w3, w4 = [random.uniform(-0.5, 0.5) for _ in range(4)]
w5, w6, w7, w8 = [random.uniform(-0.5, 0.5) for _ in range(4)]
b1, b2 = 0.5, 0.7
x1, x2 = 0.05, 0.10
h1 = tanh(w1 * x1 + w2 * x2 + b1)
h2 = tanh(w3 * x1 + w4 * x2 + b1)
output1 = tanh(w5 * h1 + w6 * h2 + b2)
output2 = tanh(w7 * h1 + w8 * h2 + b2)
print("Output of the network:", output1, output2)
