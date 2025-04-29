import numpy as np

# Activation functions
def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# RNN Model
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wx = np.random.randn(hidden_size, input_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        hs = []
        h = h_prev
        for x in inputs:
            h = tanh(np.dot(self.Wx, x) + np.dot(self.Wh, h) + self.bh)
            hs.append(h)
        y = softmax(np.dot(self.Wy, h) + self.by)
        return y, hs

    def backward(self, inputs, hs, target, y_pred, learning_rate=0.01):
        dy = y_pred - target
        dWy = np.dot(dy, hs[-1].T)
        dby = dy

        dh = np.dot(self.Wy.T, dy)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(len(inputs))):
            dh_raw = dh * (1 - hs[t] ** 2)
            dWx += np.dot(dh_raw, inputs[t].T)
            if t > 0:
                dWh += np.dot(dh_raw, hs[t-1].T)
            dbh += dh_raw
            dh = np.dot(self.Wh.T, dh_raw)

        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        loss = -np.sum(target * np.log(y_pred + 1e-8))
        return loss

# Prepare data
def prepare_data(text):
    words = text.lower().split()
    if len(words) != 4:
        raise ValueError("Please provide exactly 4 words.")

    vocab = list(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    def one_hot(idx, vocab_size):
        vec = np.zeros((vocab_size, 1))
        vec[idx] = 1
        return vec

    inputs = [one_hot(word_to_idx[word], len(vocab)) for word in words[:3]]
    target = one_hot(word_to_idx[words[3]], len(vocab))

    return inputs, target, vocab, word_to_idx, idx_to_word

# Main
if __name__ == "__main__":
    text = "My name is Mohammed"  # <-- الجملة المطلوبة
    inputs, target, vocab, word_to_idx, idx_to_word = prepare_data(text)

    rnn = SimpleRNN(input_size=len(vocab), hidden_size=5, output_size=len(vocab))
    h_prev = np.zeros((5, 1))

    # Training
    for epoch in range(1000):
        y_pred, hs = rnn.forward(inputs, h_prev)
        loss = rnn.backward(inputs, hs, target, y_pred)
        # No printing during training

    # Testing
    y_pred, _ = rnn.forward(inputs, h_prev)
    pred_idx = np.argmax(y_pred)
    pred_word = idx_to_word[pred_idx]

    print("\n--- Result ---")
    print(f"Input words: {' '.join(text.split()[:3])}")
    print(f"Actual 4th word: {text.split()[3]}")
    print(f"Predicted 4th word: {pred_word}")