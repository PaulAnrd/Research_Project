import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

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

        scale = 0.01
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_i = np.zeros((hidden_size, 1))

        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_c = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_o = np.zeros((hidden_size, 1))

        self.W_y = np.random.randn(output_size, hidden_size) * scale
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x, h_prev=None, c_prev=None):
        T = len(x)
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))
        if c_prev is None:
            c_prev = np.zeros((self.hidden_size, 1))
        h_all = [h_prev]
        c_all = [c_prev]
        y_all = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            z = np.concatenate((h_all[-1], x_t), axis=0)

            f_t = sigmoid(np.dot(self.W_f, z) + self.b_f)
            i_t = sigmoid(np.dot(self.W_i, z) + self.b_i)
            c_tilde = tanh(np.dot(self.W_c, z) + self.b_c)
            c_t = f_t * c_all[-1] + i_t * c_tilde
            o_t = sigmoid(np.dot(self.W_o, z) + self.b_o)
            h_t = o_t * tanh(c_t)
            y_t = np.dot(self.W_y, h_t) + self.b_y

            h_all.append(h_t)
            c_all.append(c_t)
            y_all.append(y_t)
        return y_all, h_all, c_all

    def backward(self, x, y_all, h_all, c_all, targets):
        T = len(x)
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_c = np.zeros_like(self.W_c)
        dW_o = np.zeros_like(self.W_o)
        dW_y = np.zeros_like(self.W_y)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_c = np.zeros_like(self.b_c)
        db_o = np.zeros_like(self.b_o)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        clip_value = 5  

        for t in reversed(range(T)):
            x_t = x[t].reshape(-1, 1)
            h_prev = h_all[t]
            c_prev = c_all[t]
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
            c_act = tanh(c_t)
            dc = dh * o_t * tanh_derivative(c_act) + dc_next

            dc_prev = dc * f_t
            df = dc * c_prev * sigmoid_derivative(f_t)
            di = dc * c_tilde * sigmoid_derivative(i_t)
            d_c_tilde = dc * i_t * tanh_derivative(c_tilde)
            do = dh * c_act * sigmoid_derivative(o_t)

            df = np.clip(df, -clip_value, clip_value)
            di = np.clip(di, -clip_value, clip_value)
            d_c_tilde = np.clip(d_c_tilde, -clip_value, clip_value)
            do = np.clip(do, -clip_value, clip_value)

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

    def clip_gradients(self, gradients, threshold=1.0):
        total_norm = 0.0
        for g in gradients:
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > threshold:
            scale = threshold / total_norm
            gradients = tuple(g * scale for g in gradients)
        return gradients

    def update_parameters(self, gradients, learning_rate, lambda_reg=1e-4):
        dW_f, dW_i, dW_c, dW_o, dW_y, db_f, db_i, db_c, db_o, db_y = gradients

        self.W_f -= learning_rate * (dW_f + lambda_reg * self.W_f)
        self.W_i -= learning_rate * (dW_i + lambda_reg * self.W_i)
        self.W_c -= learning_rate * (dW_c + lambda_reg * self.W_c)
        self.W_o -= learning_rate * (dW_o + lambda_reg * self.W_o)
        self.W_y -= learning_rate * (dW_y + lambda_reg * self.W_y)

        self.b_f -= learning_rate * db_f
        self.b_i -= learning_rate * db_i
        self.b_c -= learning_rate * db_c
        self.b_o -= learning_rate * db_o
        self.b_y -= learning_rate * db_y

    def train(self, X, targets, learning_rate=1e-7, epochs=100):
        num_samples = len(X)
        for epoch in range(epochs):
            total_loss = 0
            for i in range(num_samples):
                x_step = [X[i]]         
                target_step = [targets[i]]
                y_all, h_all, c_all = self.forward(x_step)
                gradients = self.backward(x_step, y_all, h_all, c_all, target_step)
                gradients = self.clip_gradients(gradients, threshold=1.0)
                self.update_parameters(gradients, learning_rate, lambda_reg=1e-4)
                loss = np.mean((y_all[0] - target_step[0].reshape(y_all[0].shape)) ** 2)
                total_loss += loss
            if epoch % 10 == 0:
                avg_loss = total_loss / num_samples
                print(f"Epoch {epoch}, Loss: {avg_loss}")
