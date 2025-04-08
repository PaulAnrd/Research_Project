import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_i = np.zeros((hidden_size, 1))

        self.W_c = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_c = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_o = np.zeros((hidden_size, 1))

        self.W_y = np.random.randn(output_size, hidden_size)
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        T = len(x)
        h = np.zeros((self.hidden_size, T + 1))
        c = np.zeros((self.hidden_size, T + 1))
        y = np.zeros((self.output_size, T))

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            h_prev = h[:, t].reshape(-1, 1)
            c_prev = c[:, t].reshape(-1, 1)

            z = np.concatenate((h_prev, x_t), axis=0)

            f_t = sigmoid(np.dot(self.W_f, z) + self.b_f)
            i_t = sigmoid(np.dot(self.W_i, z) + self.b_i)
            c_tilde = tanh(np.dot(self.W_c, z) + self.b_c)
            c_t = f_t * c_prev + i_t * c_tilde
            o_t = sigmoid(np.dot(self.W_o, z) + self.b_o)
            h_t = o_t * tanh(c_t)
            y_t = np.dot(self.W_y, h_t) + self.b_y

            h[:, t + 1] = h_t.reshape(-1)
            c[:, t + 1] = c_t.reshape(-1)
            y[:, t] = y_t.reshape(-1)

        return y, h, c

    def backward(self, x, y, h, c, targets):
        T = len(x)
        dW_f, dW_i, dW_c, dW_o, dW_y = (np.zeros_like(self.W_f), np.zeros_like(self.W_i),
                                         np.zeros_like(self.W_c), np.zeros_like(self.W_o),
                                         np.zeros_like(self.W_y))
        db_f, db_i, db_c, db_o, db_y = (np.zeros_like(self.b_f), np.zeros_like(self.b_i),
                                         np.zeros_like(self.b_c), np.zeros_like(self.b_o),
                                         np.zeros_like(self.b_y))

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x_t = x[t].reshape(-1, 1)
            h_prev = h[:, t].reshape(-1, 1)
            c_prev = c[:, t].reshape(-1, 1)

            z = np.concatenate((h_prev, x_t), axis=0)

            f_t = sigmoid(np.dot(self.W_f, z) + self.b_f)
            i_t = sigmoid(np.dot(self.W_i, z) + self.b_i)
            c_tilde = tanh(np.dot(self.W_c, z) + self.b_c)
            c_t = f_t * c_prev + i_t * c_tilde
            o_t = sigmoid(np.dot(self.W_o, z) + self.b_o)
            h_t = o_t * tanh(c_t)
            y_t = np.dot(self.W_y, h_t) + self.b_y

            dy = y_t - targets[t].reshape(-1, 1)
            dW_y += np.dot(dy, h_t.T)
            db_y += dy

            dh = np.dot(self.W_y.T, dy) + dh_next
            dc = dh * o_t * tanh_derivative(tanh(c_t)) + dc_next

            dc_prev = dc * f_t
            df = dc * c_prev * sigmoid_derivative(f_t)
            di = dc * c_tilde * sigmoid_derivative(i_t)
            d_c_tilde = dc * i_t * tanh_derivative(c_tilde)
            do = dh * tanh(c_t) * sigmoid_derivative(o_t)

            dW_f += np.dot(df, z.T)
            dW_i += np.dot(di, z.T)
            dW_c += np.dot(d_c_tilde, z.T)
            dW_o += np.dot(do, z.T)

            db_f += df
            db_i += di
            db_c += d_c_tilde
            db_o += do

            dz = (np.dot(self.W_f.T, df) +
                  np.dot(self.W_i.T, di) +
                  np.dot(self.W_c.T, d_c_tilde) +
                  np.dot(self.W_o.T, do))

            dh_next = dz[:self.hidden_size, :]
            dc_next = dc_prev

        return dW_f, dW_i, dW_c, dW_o, dW_y, db_f, db_i, db_c, db_o, db_y

    def update_parameters(self, gradients, learning_rate):
        dW_f, dW_i, dW_c, dW_o, dW_y, db_f, db_i, db_c, db_o, db_y = gradients

        self.W_f -= learning_rate * dW_f
        self.W_i -= learning_rate * dW_i
        self.W_c -= learning_rate * dW_c
        self.W_o -= learning_rate * dW_o
        self.W_y -= learning_rate * dW_y

        self.b_f -= learning_rate * db_f
        self.b_i -= learning_rate * db_i
        self.b_c -= learning_rate * db_c
        self.b_o -= learning_rate * db_o
        self.b_y -= learning_rate * db_y

    def train(self, X, targets, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            y, h, c = self.forward(X)
            gradients = self.backward(X, y, h, c, targets)
            self.update_parameters(gradients, learning_rate)
            loss = np.mean((y - targets) ** 2)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')





#--------------------------------------------------------------


