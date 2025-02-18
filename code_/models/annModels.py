import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x > 0, 1, 0)

class MLP:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        self.W1 = np.random.randn(n_input, n_hidden1) * np.sqrt(2. / n_input)
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2. / n_hidden1)
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = np.random.randn(n_hidden2, n_output) * np.sqrt(2. / n_hidden2)
        self.b3 = np.zeros((1, n_output))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3
        return self.a3

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]

        d_loss_a3 = (output - y) / m
        d_z3 = d_loss_a3
        dW3 = np.dot(self.a2.T, d_z3)
        db3 = np.sum(d_z3, axis=0, keepdims=True)

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * relu_deriv(self.z2)
        dW2 = np.dot(self.a1.T, d_z2)
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * relu_deriv(self.z1)
        dW1 = np.dot(X.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses